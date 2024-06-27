#ifndef CONFIG_HPP
#define CONFIG_HPP

namespace parallelisation {

enum class direction { x, y, z };

enum class FluidType { isothermal, adiabatic, none };

enum class BoundaryType { constant_extrapolation };

} // namespace parallelisation

#endif