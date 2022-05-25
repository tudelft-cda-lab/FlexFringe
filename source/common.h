// this file contains helper functions and macros to be
// included elsewhere

#ifndef _COMMON_H_
#define _COMMON_H_

extern bool debugging_enabled;
extern char* gitversion;

#define DEBUG(x) do { \
  if (debugging_enabled) { std::cerr << x << std::endl; } \
} while (0)

#endif 
