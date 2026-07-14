# Java

# Find Java
find_package( Java 1.8 COMPONENTS Development )

if( NOT Java_FOUND )
  message( "JAVA JDK NOT found - skipping Gpufit Java binding!" )
  return()
endif()

# Find JNI
find_package( JNI )

if( NOT JNI_FOUND )
  message( "JAVA JNI NOT found - skipping Gpufit Java binding!" )
  return()
endif()

# Set the current process environment variable JAVA_HOME to ensure that only one version 
# of Java is used
get_filename_component(JAVA_BIN_DIR ${Java_JAVAC_EXECUTABLE} PATH)
get_filename_component(JAVA_HOME ${JAVA_BIN_DIR} PATH)
set( ENV{JAVA_HOME} ${JAVA_HOME} )

# Try running Java
execute_process( COMMAND java -version RESULT_VARIABLE java_return_value )

if( java_return_value EQUAL -1 )
    message( "Java NOT found by environment - skipping Gpufit Java binding!" )
    return()
endif()

# Adapter library

add_subdirectory( adapter )

# Java package

set( build_directory "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/java" )
set( java_directory "${CMAKE_CURRENT_SOURCE_DIR}/gpufit" )
set( copy_java_jars_script "${CMAKE_CURRENT_BINARY_DIR}/copy_java_jars.cmake" )
file( WRITE "${copy_java_jars_script}" [=[
if( NOT DEFINED java_directory OR NOT DEFINED build_directory )
  message( FATAL_ERROR "java_directory and build_directory must be set" )
endif()

file( GLOB java_jars
  "${java_directory}/build/libs/Gpufit*.jar"
  "${java_directory}/build/libs/gpufit*.jar"
  LIST_DIRECTORIES false
)

if( NOT java_jars )
  message( FATAL_ERROR "No Java JAR found in ${java_directory}/build/libs after Gradle build." )
endif()

file( COPY ${java_jars} DESTINATION "${build_directory}" )
]=] )

set( binary_files
  "$<TARGET_FILE:Gpufit>"
  "$<TARGET_FILE:GpufitJNI>"
  "${CMAKE_CURRENT_SOURCE_DIR}/README.txt"
)

if ( USE_CUBLAS AND DEFINED CUBLAS_DLL )
	set( binary_files ${binary_files} ${CUBLAS_DLL} )
endif()

if ( WIN32 )
    set( gradle_command cmd /c gradlew.bat build --no-daemon "-Dorg.gradle.java.home=${JAVA_HOME}" )
else()
    set( gradle_command ./gradlew build --no-daemon )
endif()

add_custom_target( JAVA_PACKAGE ALL
  COMMAND ${CMAKE_COMMAND} -E
    remove_directory ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    make_directory ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    chdir ${java_directory} ${gradle_command}
  COMMAND ${CMAKE_COMMAND}
    -Djava_directory=${java_directory}
    -Dbuild_directory=${build_directory}
    -P ${copy_java_jars_script}
  COMMAND ${CMAKE_COMMAND} -E
    copy_if_different ${binary_files} ${build_directory}
  COMMAND ${CMAKE_COMMAND} -E
    copy_directory "${java_directory}/src/test/" ${build_directory}
)
set_property( TARGET JAVA_PACKAGE PROPERTY FOLDER CMakePredefinedTargets )
add_dependencies( JAVA_PACKAGE Gpufit GpufitJNI )