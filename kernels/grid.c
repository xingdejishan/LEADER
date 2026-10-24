/* A bounded-distance map lookup for the arbitration.
 *
 * The verification only needs, for every point of the frame's own scan, the distance to the
 * nearest allowed map point -- and only up to 1.0 m.  Beyond that the observation likelihood is
 * flat to within 1e-3, because its Gaussian is already 5e-4 at one metre while the uniform floor
 * is 0.05, and every overlap threshold the decision reads is at or below 1.0 m.
 *
 * So the cell side is 1.0 m, which makes the 27-cell probe provably contain every map point
 * within 1.0 m, and a point whose nearest allowed neighbour is farther is reported as ">= 1.0".
 * The map is laid out by counting sort into a dense cell array, so a lookup is an offset and a
 * short scan instead of a tree descent.
 *
 * Per-candidate exclusion is preserved exactly: every map point carries the reference that owns
 * it, and the caller passes an allowed-reference mask per pose.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int32_t dim[3];
    double origin[3];
    double cell;
    double cell2;
    int64_t ncells;
    int32_t *start;        /* ncells + 1, prefix sums */
    int32_t *cursor;       /* ncells, scratch */
    int32_t *order;        /* npoints, original indices grouped by cell */
    double *sxyz;          /* npoints, coordinates REORDERED into cell order */
    int32_t *slabel;       /* npoints, owners reordered with them */
    const double *xyz;     /* borrowed, 3 per point */
    const int32_t *label;  /* borrowed, 1 per point */
    int64_t npoints;
    int64_t kept;
} grid_t;

static int64_t cell_index(const grid_t *g, int cx, int cy, int cz)
{
    return ((int64_t)cx * g->dim[1] + cy) * g->dim[2] + cz;
}

grid_t *grid_build(const double *xyz, const int32_t *label, int64_t n,
                   const double *lo, const double *hi, double cell)
{
    grid_t *g = (grid_t *)calloc(1, sizeof(grid_t));
    if (!g) return NULL;
    g->cell = cell;
    g->cell2 = cell * cell;
    g->xyz = xyz;
    g->label = label;
    g->npoints = n;
    for (int a = 0; a < 3; a++) {
        g->origin[a] = lo[a] - cell;                    /* one empty cell of border */
        int32_t d = (int32_t)ceil((hi[a] - g->origin[a]) / cell) + 3;
        g->dim[a] = d < 4 ? 4 : d;
    }
    g->ncells = (int64_t)g->dim[0] * g->dim[1] * g->dim[2];
    g->start = (int32_t *)calloc((size_t)g->ncells + 1, sizeof(int32_t));
    g->cursor = (int32_t *)calloc((size_t)g->ncells, sizeof(int32_t));
    g->order = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n > 0 ? n : 1));
    /* the scatter also lays the coordinates out in cell order, so the query walks contiguous
     * memory per cell instead of gathering random addresses out of the original cloud */
    g->sxyz = (double *)malloc(sizeof(double) * 3 * (size_t)(n > 0 ? n : 1));
    g->slabel = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n > 0 ? n : 1));
    if (!g->start || !g->cursor || !g->order || !g->sxyz || !g->slabel) { return NULL; }

    for (int64_t i = 0; i < n; i++) {
        int cx = (int)((xyz[3 * i + 0] - g->origin[0]) / cell);
        int cy = (int)((xyz[3 * i + 1] - g->origin[1]) / cell);
        int cz = (int)((xyz[3 * i + 2] - g->origin[2]) / cell);
        if (cx < 0 || cy < 0 || cz < 0 ||
            cx >= g->dim[0] || cy >= g->dim[1] || cz >= g->dim[2]) continue;
        g->start[cell_index(g, cx, cy, cz) + 1]++;
    }
    for (int64_t i = 1; i <= g->ncells; i++) g->start[i] += g->start[i - 1];
    memcpy(g->cursor, g->start, sizeof(int32_t) * (size_t)g->ncells);
    for (int64_t i = 0; i < n; i++) {
        int cx = (int)((xyz[3 * i + 0] - g->origin[0]) / cell);
        int cy = (int)((xyz[3 * i + 1] - g->origin[1]) / cell);
        int cz = (int)((xyz[3 * i + 2] - g->origin[2]) / cell);
        if (cx < 0 || cy < 0 || cz < 0 ||
            cx >= g->dim[0] || cy >= g->dim[1] || cz >= g->dim[2]) continue;
        int64_t idx = cell_index(g, cx, cy, cz);
        int32_t pos = g->cursor[idx]++;
        g->order[pos] = (int32_t)i;
        g->sxyz[3 * pos + 0] = xyz[3 * i + 0];
        g->sxyz[3 * pos + 1] = xyz[3 * i + 1];
        g->sxyz[3 * pos + 2] = xyz[3 * i + 2];
        g->slabel[pos] = label[i];
        g->kept++;
    }
    return g;
}

/* nearest allowed map point, reported exactly up to one cell side, else 1.0001 */
void grid_query(const grid_t *g, const double *q, int64_t nq,
                const uint8_t *allowed, double *out)
{
    const int32_t d0 = g->dim[0], d1 = g->dim[1], d2 = g->dim[2];
    const double cell = g->cell, cell2 = g->cell2;
    for (int64_t i = 0; i < nq; i++) {
        const double qx = q[3 * i + 0], qy = q[3 * i + 1], qz = q[3 * i + 2];
        const int cx = (int)((qx - g->origin[0]) / cell);
        const int cy = (int)((qy - g->origin[1]) / cell);
        const int cz = (int)((qz - g->origin[2]) / cell);
        double best2 = 1e300;
        for (int dx = -1; dx <= 1; dx++) {
            const int jx = cx + dx;
            if (jx < 0 || jx >= d0) continue;
            for (int dy = -1; dy <= 1; dy++) {
                const int jy = cy + dy;
                if (jy < 0 || jy >= d1) continue;
                for (int dz = -1; dz <= 1; dz++) {
                    const int jz = cz + dz;
                    if (jz < 0 || jz >= d2) continue;
                    const int64_t idx = cell_index(g, jx, jy, jz);
                    const int32_t k0 = g->start[idx], k1 = g->start[idx + 1];
                    for (int32_t k = k0; k < k1; k++) {
                        if (!allowed[g->slabel[k]]) continue;
                        const double ax = g->sxyz[3 * k + 0] - qx;
                        const double ay = g->sxyz[3 * k + 1] - qy;
                        const double az = g->sxyz[3 * k + 2] - qz;
                        const double dd = ax * ax + ay * ay + az * az;
                        if (dd < best2) best2 = dd;
                    }
                }
            }
        }
        out[i] = (best2 <= cell2) ? sqrt(best2) : 1.0001;
    }
}

void grid_free(grid_t *g)
{
    if (!g) return;
    free(g->start);
    free(g->cursor);
    free(g->order);
    free(g->sxyz);
    free(g->slabel);
    free(g);
}

int64_t grid_cells(const grid_t *g) { return g->ncells; }
int64_t grid_kept(const grid_t *g) { return g->kept; }
