cmake_minimum_required(VERSION 3.5)

get_filename_component(PROJNAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
Project(${PROJNAME})
Message(STATUS "-------------------------------")
Message(STATUS "Processing Project ${PROJNAME}:")

#####################################################################################
# look for nvpro_core 1) as a sub-folder 2) at some other locations
# this cannot be put anywhere else since we still didn't find setup.cmake yet
#
if(NOT BASE_DIRECTORY)

  find_path(BASE_DIRECTORY
    NAMES nvpro_core/cmake/setup.cmake
    PATHS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../.. 
    REQUIRED
    DOC "Directory containing nvpro_core"
    )
endif()
if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
else()
  message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing nvpro_core")
endif()
    
_add_project_definitions(${PROJNAME})


#####################################################################################
# additions from packages needed for this sample
# add refs  in LIBRARIES_OPTIMIZED
# add refs  in LIBRARIES_DEBUG
# add files in PACKAGE_SOURCE_FILES

_add_package_VulkanSDK()
_add_package_IMGUI()
_add_package_ShaderC()

#####################################################################################
_find_package_OpenMP()
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#####################################################################################
# process the rest of some cmake code that needs to be done *after* the packages add
_add_nvpro_core_lib()

add_definitions(-DCSF_ZIP_SUPPORT=1)

#####################################################################################
# Source files for this project
#
file(GLOB SOURCE_FILES *.cpp *.hpp *.inl *.h *.c)
file(GLOB GLSL_FILES *.glsl)


#####################################################################################
# Executable
#
if(WIN32 AND NOT GLUT_FOUND)
  add_definitions(/wd4996) #remove printf warning
  add_definitions(/wd4244) #remove double to float conversion warning
  add_definitions(/wd4305) #remove double to float truncation warning
else()
  add_definitions(-fpermissive)
endif()
add_executable(${PROJNAME} ${SOURCE_FILES} ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES} ${GLSL_FILES})

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJNAME})

#####################################################################################
# common source code needed for this sample
#
source_group(common FILES 
  ${COMMON_SOURCE_FILES}
  ${PACKAGE_SOURCE_FILES}
)
source_group(shaders FILES 
  ${GLSL_FILES}
)
source_group("Source Files" FILES ${SOURCE_FILES})

if(UNIX)
  set(UNIXLINKLIBS dl pthread)
else()
  set(UNIXLINKLIBS)
endif()

#####################################################################################
# Linkage
#
target_link_libraries(${PROJNAME} ${PLATFORM_LIBRARIES} nvpro_core)

foreach(DEBUGLIB ${LIBRARIES_DEBUG})
  target_link_libraries(${PROJNAME} debug ${DEBUGLIB})
endforeach(DEBUGLIB)

foreach(RELEASELIB ${LIBRARIES_OPTIMIZED})
  target_link_libraries(${PROJNAME} optimized ${RELEASELIB})
endforeach(RELEASELIB)

#####################################################################################
# copies binaries that need to be put next to the exe files (ZLib, etc.)
#

_finalize_target( ${PROJNAME} )

LIST(APPEND GLSL_FILES "common.h")
install(FILES ${GLSL_FILES} CONFIGURATIONS Release DESTINATION "bin_${ARCH}/GLSL_${PROJNAME}")
install(FILES ${GLSL_FILES} CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug/GLSL_${PROJNAME}")
