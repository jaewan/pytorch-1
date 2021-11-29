set(ATen_HIP_INCLUDE ${ATen_HIP_INCLUDE}
  "${CMAKE_CURRENT_SOURCE_DIR}"
PARENT_SCOPE)

set(ATen_HIP_SRCS ${ATen_HIP_SRCS}
  ${CMAKE_CURRENT_SOURCE_DIR}/THHStorage.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THHTensor.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/THHSleep.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/THHStorage.hip
  PARENT_SCOPE)

install(FILES
          THH.h
          THHGeneral.h
          THHGeneral.hpp
          THHSleep.h
          THHStorage.h
          THHTensor.h
          THHAtomics.cuh
          THHGenerateByteType.h
          # See Note [TH abstraction violation]
          THHTensor.hpp
          THHStorage.hpp
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THH")

install(FILES
          generic/THHStorage.cpp
          generic/THHStorage.hip
          generic/THHStorage.h
          generic/THHTensor.cpp
          generic/THHTensor.h
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THH/generic")
