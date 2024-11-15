#ifndef __BMARS_FLAGS_H
#define __BMARS_FLAGS_H

// #define OUTLIER_MAX 1000.f

/*------------------------------- LOGICAL FLAGS ------------------------------*/

/* Number of iterations where classic bidir is used instead of MARS */
#define TRAIN_ITERATIONS 3

/* Number of iterations after which the SPP is doubled in MARS */
#define M_DOUBLE_TIME 8

#define INITIAL_EARS_FACTOR 1e-3

/* If set, the final image will be the variance-weighted sum of all train images */
#define USE_VARWEIGHTED_BITMAP

#define LOW_DISCREPANCY_NUM_SAMPLES

/*-------------------------------- DEBUG FLAGS -------------------------------*/

// #define MARS_INCLUDE_AOVS

/* If set, verifies the computation of the splitting factors by recomputing the known
   ones and comparing them and the corresponding bsdf weights */
// #define CHECK_SPLIT_FACTOR_CALC

// #define BMARS_DEBUG_FP
// #define MTS_DEBUG_FP

#endif // __BMARS_FLAGS_H
