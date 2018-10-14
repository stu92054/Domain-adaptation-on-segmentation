include_directories( ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
link_directories( ${Boost_LIBRARY_DIR} )
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/__init__.py" "" )

add_library( ccnn SHARED boost.cpp ccnn.cpp util.cpp constraintloss.cpp )
target_link_libraries( ccnn ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} constraintloss util)

set_target_properties( ccnn PROPERTIES PREFIX "")
if(APPLE)
	set_target_properties( ccnn PROPERTIES SUFFIX ".so" )
endif()
