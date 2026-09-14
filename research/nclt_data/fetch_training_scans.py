import hashlib
import io
import json
import shlex
from pathlib import Path, PurePosixPath
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

import paramiko


REMOTE = '/root/rivermind-data/datasets/NCLT'


def connect(password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('sc01-ssh.gpuhome.cc', port=30826, username='root', password=password,
                   look_for_keys=False, allow_agent=False, timeout=15)
    client.get_transport().set_keepalive(15)
    return client


def resume_fetch(password, workspace):
    workspace = Path(workspace)
    reference = Path(r'\\wsl.localhost\Ubuntu\home\zhang\rscore-l-local\data\manifest.json')
    rows = json.loads(reference.read_text())['train']
    assert len(rows) == 907 and len({r['frame_id'] for r in rows}) == 907
    audit = workspace / 'glace-local/data/train907_raw_scan_audit'
    destination = workspace / 'glace-local/data/scans'
    audit.mkdir(parents=True, exist_ok=True)
    inventory = audit / 'server_inventory.json'
    if not inventory.exists():
        code = '''import pathlib,json,hashlib
root=pathlib.Path(ROOT)
records=[]
for row in ROWS:
 relative=row['session_id']+'/velodyne_sync/'+row['frame_id']+'.bin'
 path=root/relative
 record=dict(row,relative_path=relative,remote_path=str(path),exists=path.is_file())
 if path.is_file():
  data=path.read_bytes()
  record.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 records.append(record)
print(json.dumps(records))
'''.replace('ROOT', repr(REMOTE)).replace('ROWS', repr([dict(frame_id=r['frame_id'], session_id=r['session_id']) for r in rows]))
        client = connect(password)
        try:
            stdin, stdout, stderr = client.exec_command('python3 -', timeout=120)
            stdin.write(code)
            stdin.channel.shutdown_write()
            records = json.loads(stdout.read())
            if stdout.channel.recv_exit_status() != 0:
                raise RuntimeError(stderr.read().decode())
            inventory.write_text(json.dumps(records, indent=2))
        finally:
            client.close()
    records = json.loads(inventory.read_text())
    expected = {r['relative_path']: r for r in records if r['exists']}
    assert {r['frame_id'] for r in records} == {r['frame_id'] for r in rows}

    def install(name, data):
        record = expected[name]
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Scan failed SHA256 check: '+name)
        if not data or len(data) % 8:
            raise ValueError('Invalid NCLT scan size')
        target = destination.joinpath(*PurePosixPath(name).parts)
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError('Invalid destination path')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('Conflicting local scan')
            return
        pending = target.with_suffix('.pending')
        pending.write_bytes(data)
        pending.replace(target)

    for archive in sorted(audit.glob('part*/*.partial')):
        recovered = 0
        try:
            with tarfile.open(archive, 'r|gz') as tar:
                for member in tar:
                    if member.name not in expected or not member.isfile():
                        raise ValueError('Unexpected partial archive member')
                    data = tar.extractfile(member).read()
                    install(member.name, data)
                    recovered += 1
        except (EOFError, tarfile.ReadError):
            pass
        print(json.dumps(dict(recovered_archive=str(archive), verified_members=recovered)), flush=True)
    remaining = [r for name, r in expected.items() if not destination.joinpath(*PurePosixPath(name).parts).exists()]
    print(json.dumps(dict(already_present=len(expected)-len(remaining), remaining=len(remaining))), flush=True)

    def worker(index):
        assigned = remaining[index::4]
        for first in range(0, len(assigned), 20):
            batch = assigned[first:first+20]
            for attempt in range(6):
                pending = [r for r in batch if not destination.joinpath(*PurePosixPath(r['relative_path']).parts).exists()]
                if not pending:
                    break
                client = None
                try:
                    client = connect(password)
                    command = 'tar -czf - -C '+shlex.quote(REMOTE)+' -- '+' '.join(shlex.quote(r['relative_path']) for r in pending)
                    _, stdout, stderr = client.exec_command(command, timeout=90)
                    payload = stdout.read()
                    if stdout.channel.recv_exit_status() != 0:
                        raise OSError('Batch SSH transfer did not finish')
                    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as tar:
                        for member in tar:
                            if member.name not in {r['relative_path'] for r in pending} or not member.isfile():
                                raise ValueError('Unexpected batch archive member')
                            install(member.name, tar.extractfile(member).read())
                    print(json.dumps(dict(worker=index, verified=min(first+20,len(assigned)), assigned=len(assigned))), flush=True)
                except (OSError, EOFError, paramiko.SSHException, tarfile.ReadError):
                    if attempt == 5:
                        raise
                    time.sleep(1)
                finally:
                    if client is not None:
                        client.close()
            if any(not destination.joinpath(*PurePosixPath(r['relative_path']).parts).exists() for r in batch):
                raise RuntimeError('Batch incomplete after retries')
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(worker, range(4)))
    for record in records:
        if record['exists']:
            target = destination.joinpath(*PurePosixPath(record['relative_path']).parts)
            assert hashlib.sha256(target.read_bytes()).hexdigest() == record['sha256']
            record.update(local_path=str(target), verified=True)
    summary = dict(remote_root=REMOTE, requested=907, verified=len(expected), records=records,
        missing=[r for r in records if not r['exists']], total_bytes=sum(r['bytes'] for r in expected.values()),
        reference_manifest_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        pairing='Exact timestamps only; missing files not replaced; original 907 image rows preserved')
    (audit / 'manifest.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k:summary[k] for k in ['requested','verified','total_bytes','missing']}, indent=2), flush=True)
    return summary




if __name__ == '__main__':
    import argparse
    import getpass
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    resume_fetch(getpass.getpass('SSH password: '), args.workspace)
