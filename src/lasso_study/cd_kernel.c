/* One cyclic coordinate-descent sweep for
 * ||Xw-y||^2/(2n) + lambda*||w||_1.
 * X is column-major (Fortran order); residual equals y-Xw on entry.
 * No allocation, threading, solver library, or implicit normalization.
 */
#include <stdint.h>

void cd_sweep(const double *X, double *residual, double *w,
              const double *column_norms, int64_t n, int64_t p,
              double lambda) {
    for (int64_t j = 0; j < p; ++j) {
        const double norm = column_norms[j];
        if (norm == 0.0) {
            w[j] = 0.0;
            continue;
        }
        const double *column = X + n * j;
        double product = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            product += column[i] * residual[i];
        }
        const double partial = product / (double)n + norm * w[j];
        double new_weight = 0.0;
        if (partial > lambda) {
            new_weight = (partial - lambda) / norm;
        } else if (partial < -lambda) {
            new_weight = (partial + lambda) / norm;
        }
        const double difference = new_weight - w[j];
        if (difference != 0.0) {
            for (int64_t i = 0; i < n; ++i) {
                residual[i] -= column[i] * difference;
            }
        }
        w[j] = new_weight;
    }
}
