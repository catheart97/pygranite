#pragma once

/**
 * @file Config.h
 * 
 * This file is used for build configuration pragmas, such as compiling LOG and LOGH Messages into 
 * the code.
 */

////////////
/// LOGGING
////////////

/// LOG is used for cpu side logging (uses cout internally so concatenate using "<<")
/// LOGG is used for gpu side logging (similar syntax as printf)
/// LOGE is used to create a logging environment (all contents will be removed when disabled)

// #define MY_LOGGING // logs only messages created with LOG, LOGG, LOGE, SLOG
// #define MY_VLOGGING // logs messages created with LOG, LOGG, LOGE, SLOG, VLOG, VLOGG, VLOGE
