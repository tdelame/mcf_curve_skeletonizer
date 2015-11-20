/* Created on: 2015-11-17
 *     Author: T.Delame (tdelame@gmail.com)
 */
# ifndef PROJECT_PROJECT_H_
# define PROJECT_PROJECT_H_
# include <float.h>
# include <iostream>
# include <ctime>
# define REAL_MIN DBL_MIN
# define REAL_MAX DBL_MAX
# if defined ( _WIN32 ) || defined( _WIN64 )
#   define PROJECT_API __declspec( dllexport )
# else
#   ifdef __APPLE__
#       define PROJECT_API
#   else
#        define PROJECT_API
#   endif
# endif
// to avoid name conflicts, all the definition are included in a global namespace
# define PROJECT_NAMESPACE mcf_curve_skeletonizer
# define BEGIN_PROJECT_NAMESPACE namespace PROJECT_NAMESPACE {
# define END_PROJECT_NAMESPACE }
BEGIN_PROJECT_NAMESPACE
typedef double real;
// basic log utility
enum severity_level {
  trace, debug, info, warning, error, fatal
};
extern const std::string severity_names[6];
# define LOG( level, message )                                              \
{                                                                           \
  if( severity_level::level > severity_level::warning )                     \
  {                                                                         \
    auto time = std::time( nullptr );                                       \
    char mbstr[ 30 ];                                                       \
    std::strftime(mbstr, sizeof(mbstr), "%c", std::localtime(&time));       \
    std::cout<< '[' << mbstr << ']'                                         \
             << severity_names[severity_level::level]<< " "                 \
             << message                                                     \
             << " (in "<< __FILE__<<":"<< __LINE__ <<")"                    \
             << std::endl;                                                  \
  }                                                                         \
  else                                                                      \
  {                                                                         \
    auto time = std::time( nullptr );                                       \
    char mbstr[ 30 ];                                                       \
    std::strftime(mbstr, sizeof(mbstr), "%c", std::localtime(&time));       \
    std::cout<< '[' << mbstr << ']'                                         \
             << severity_names[severity_level::level]<< " "                 \
             << message << std::endl;                                       \
  }                                                                         \
}
END_PROJECT_NAMESPACE
# endif
