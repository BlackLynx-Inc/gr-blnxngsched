if(NOT PKG_CONFIG_FOUND)
    INCLUDE(FindPkgConfig)
endif()
PKG_CHECK_MODULES(PC_BLNXNGSCHED blnxngsched)

FIND_PATH(
    BLNXNGSCHED_INCLUDE_DIRS
    NAMES blnxngsched/api.h
    HINTS $ENV{BLNXNGSCHED_DIR}/include
        ${PC_BLNXNGSCHED_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    BLNXNGSCHED_LIBRARIES
    NAMES gnuradio-blnxngsched
    HINTS $ENV{BLNXNGSCHED_DIR}/lib
        ${PC_BLNXNGSCHED_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/blnxngschedTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(BLNXNGSCHED DEFAULT_MSG BLNXNGSCHED_LIBRARIES BLNXNGSCHED_INCLUDE_DIRS)
MARK_AS_ADVANCED(BLNXNGSCHED_LIBRARIES BLNXNGSCHED_INCLUDE_DIRS)
