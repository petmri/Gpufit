
# Cpufit

set( CpuHeaders
	cpufit.h
	../Gpufit/constants.h
	info.h
	lm_fit.h
	interface.h
	patlak_linear.h
)

set( CpuSources
	cpufit.cpp
	info.cpp
	lm_fit.cpp
	lm_fit_cpp.cpp
	interface.cpp
	patlak_linear.cpp
	Cpufit.def
)

add_library( Cpufit SHARED
	${CpuHeaders}
	${CpuSources}
)
set_target_properties( Cpufit
	PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
		CXX_VISIBILITY_PRESET hidden
)

#install( TARGETS Cpufit RUNTIME DESTINATION bin )

add_subdirectory( matlab )
add_subdirectory( python )
