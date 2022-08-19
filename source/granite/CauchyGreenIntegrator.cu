#include "CauchyGreenIntegrator.hpp"

/*
 * ! kernel part
 */

#ifdef __NVCC__
__global__ void integrate_cauchy_green( //
    size_t N,                           // number of points, target_points is 5*N
    my::math::Vec2 * center_points,     //
    my::math::Vec2 * target_points,     //
    my::math::Mat2x2 * tensors,         //
    float d_lon,                        //
    float d_lat                         //
)
{
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t NUM_THREADS = gridDim.x * blockDim.x;

    for (size_t pid = idx; pid < N; pid += NUM_THREADS)
    {
        const auto STRIDE{5 * pid};

        const auto Z0{center_points[pid]};
        const auto Z1{target_points[STRIDE]};

        const auto A1{target_points[STRIDE + 1]};
        const auto B1{target_points[STRIDE + 2]};
        const auto C1{target_points[STRIDE + 3]};
        const auto D1{target_points[STRIDE + 4]};

        // d = [x, y; z, w]
        const double x{(double(B1.x) - double(A1.x)) / (2.0 * d_lon)};
        const double z{(double(B1.y) - double(A1.y)) / (2.0 * d_lon)};

        const double y{(double(D1.x) - double(C1.x)) / (2.0 * d_lat)};
        const double w{(double(D1.y) - double(C1.y)) / (2.0 * d_lat)};

        double alpha{1 / cos(double(Z0.y) * my::math::PI_D / 180.0)};
        double beta{my::math::sq(cos(double(Z1.y) * my::math::PI_D / 180.0))};

        MY_LOGE(if (!idx) {
            MY_LOGG("(GPU Device) Z1 Vec2 %f %f", Z1.x, Z1.y);
            MY_LOGG("(GPU Device) A1 Vec2 %f %f", A1.x, A1.y);
            MY_LOGG("(GPU Device) B1 Vec2 %f %f", B1.x, B1.y);
            MY_LOGG("(GPU Device) C1 Vec2 %f %f", C1.x, C1.y);
            MY_LOGG("(GPU Device) D1 Vec2 %f %f", D1.x, D1.y);
            MY_LOGG("(GPU Device) x: %f, y: %f, z: %f, w: %f", x, y, z, w);
        })

        double t{alpha * (beta * x * y + w * z)};
        tensors[pid] = my::math::Mat2x2{
            float(my::math::sq(alpha) * (my::math::sq(x) * beta + my::math::sq(z))), // 00
            float(t),                                                                // 01
            float(t),                                                                // 10
            float(beta * my::math::sq(y) + my::math::sq(w))                          // 11
        };
    }
}

#endif

/*
 * ! host PART
 */

granite::CauchyGreenProperties
granite::CauchyGreenIntegrator::ensureProperties(granite::CauchyGreenProperties & props) const
{
    const float min_x{(props.MIN[0] - _settings.Offset[0]) / _settings.GridScale[0]};
    const float min_y{(props.MIN[1] - _settings.Offset[1]) / _settings.GridScale[1]};
    const float max_x{(props.MAX[0] - _settings.Offset[0]) / _settings.GridScale[0]};
    const float max_y{(props.MAX[0] - _settings.Offset[1]) / _settings.GridScale[1]};

    if (min_x < 0 || min_y < 0)
        MY_USER_ERROR("Initial CauchyGreen Grid must be inside windfield domain. (MIN)");

    if (min_x >= max_x || min_y >= max_y) MY_USER_ERROR("Minimum domain is larger than max domain!");

    if (max_x > _grid.X || max_y > _grid.X)
        MY_USER_ERROR("Initial CauchyGreen Frid must be inside windfield domain. (MAX)");

    return props;
}

// todo: refactor this with the out_of_box methods
my::math::Vec2 granite::CauchyGreenIntegrator::ensureDomain(my::math::Vec2 & o, my::math::Vec2 n)
{
    const float dm[2]{_grid.X * _settings.GridScale[0], _grid.Y * _settings.GridScale[1]};

    if (_settings.SphereRadius >= 0.f)
    {
        my::math::Vec2 n_{(n.x - _settings.Offset[0]), (n.y - _settings.Offset[1])};

        if (n_.y < 0) // DOWN
        {
            my::math::Vec2 o_{(o.x - _settings.Offset[0]), (o.y - _settings.Offset[1])};

            auto tx{(-o_.y) / (n_.y - o_.y) * (n_.x - o_.x) + o_.x}; // intersection point
            n.y = -(n_.y) + _settings.Offset[1];
            n.x = fmodf(dm[0] + 180.0f - n_.x + 2.f * tx, dm[0]) + _settings.Offset[0];
        }
        else if (n_.y >= dm[1]) // UP
        {
            my::math::Vec2 o_{(o.x - _settings.Offset[0]), (o.y - _settings.Offset[1])};

            auto tx{(dm[1] - o_.y) / (n_.y - o_.y) * (n_.x - o_.x) + o_.x}; // intersection point
            n.y = (n_.y - dm[1]) + _settings.Offset[1];
            n.x = fmodf(dm[0] + 180.0f - n_.x + 2.f * tx, dm[0]) + _settings.Offset[0];
        }
    }
    else
    {
        n.y = fmodf((n.y - _settings.Offset[1] + dm[1]), dm[1]) + _settings.Offset[1];
    }

    n.x = fmodf((n.x - _settings.Offset[0] + dm[0]), dm[0]) + _settings.Offset[0];

    return n;
}

pybind11::array_t<float> granite::CauchyGreenIntegrator::computeCG()
{
    using namespace my::math;

    _cgproperties = ensureProperties(_cgproperties);

    const size_t N{_cgproperties.DIM[0] * _cgproperties.DIM[1]};
    const float dx{(_cgproperties.MAX[0] - _cgproperties.MIN[0]) / _cgproperties.DIM[0]};
    const float dy{(_cgproperties.MAX[1] - _cgproperties.MIN[1]) / _cgproperties.DIM[1]};

    _cgproperties.DLON *= _settings.GridScale[0];
    _cgproperties.DLAT *= _settings.GridScale[1];

    std::vector<my::math::Vec2> start_particles(N * 5);
    std::vector<my::math::Vec2> center_points(N);
    for (size_t i = 0; i < _cgproperties.DIM[0]; ++i)
    {
        for (size_t j = 0; j < _cgproperties.DIM[1]; ++j)
        {
            // center point
            auto center =
                my::math::Vec2{_cgproperties.MIN[0] + i * dx, _cgproperties.MIN[1] + j * dy};
            center_points[i * _cgproperties.DIM[1] + j] = center;

            start_particles[i * _cgproperties.DIM[1] * 5 + j * 5] = center;

            // x offset points
            start_particles[i * _cgproperties.DIM[1] * 5 + j * 5 + 1] =
                ensureDomain(center, center + my::math::Vec2{-_cgproperties.DLON, 0.f});
            start_particles[i * _cgproperties.DIM[1] * 5 + j * 5 + 2] =
                ensureDomain(center, center + my::math::Vec2{_cgproperties.DLON, 0.f});

            // y offset points
            start_particles[i * _cgproperties.DIM[1] * 5 + j * 5 + 3] =
                ensureDomain(center, center + my::math::Vec2{0.f, -_cgproperties.DLAT});
            start_particles[i * _cgproperties.DIM[1] * 5 + j * 5 + 4] =
                ensureDomain(center, center + my::math::Vec2{0.f, _cgproperties.DLAT});
        }
    }

    MY_LOG(MY_VAR(_cgproperties.DLON))
    MY_LOG(MY_VAR(_cgproperties.DLAT))
    // MY_LOG("(GPU Host) Z0 " << start_particles[0])
    // MY_LOG("(GPU Host) A0 " << start_particles[1])
    // MY_LOG("(GPU Host) B0 " << start_particles[2])
    // MY_LOG("(GPU Host) C0 " << start_particles[3])
    // MY_LOG("(GPU Host) D0 " << start_particles[4])

    my::math::Vec2 * _init_points_d{nullptr};
    cudaMalloc(&_init_points_d, N * sizeof(my::math::Vec2));
    MY_CUDA_ERROR_CHECK

    TrajectorySet set(Space::Space2D, N * 5);
    set.add_(start_particles);
    initSet(set);

    auto set_ = compute();

    my::math::Vec2 * center_points_d{nullptr};
    cudaMalloc(&center_points_d, sizeof(my::math::Vec2) * N);
    MY_CUDA_ERROR_CHECK

    cudaMemcpy(center_points_d, center_points.data(), N * sizeof(my::math::Vec2),
               cudaMemcpyHostToDevice);
    MY_CUDA_ERROR_CHECK

    my::math::Vec2 * target_points_d{reinterpret_cast<my::math::Vec2 *>(lastDevicePointer())};

    my::math::Mat2x2 * tensors_d{nullptr};
    cudaMalloc(&tensors_d, sizeof(my::math::Mat2x2) * N);
    MY_CUDA_ERROR_CHECK

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    const unsigned int THREADS =
        static_cast<unsigned int>(std::min(static_cast<size_t>(devProp.maxThreadsPerBlock), N));
    const unsigned int BLOCKS = static_cast<unsigned int>(std::min(
        my::math::sdiv<size_t>(THREADS, N), static_cast<size_t>(devProp.maxThreadsDim[0])));

    integrate_cauchy_green<<<BLOCKS, THREADS>>>(N,                  //
                                                center_points_d,    //
                                                target_points_d,    //
                                                tensors_d,          //
                                                _cgproperties.DLON, //
                                                _cgproperties.DLAT  //
    );

    pybind11::array_t<float> result = pybind11::array_t<float>(N * 4);

    pybind11::buffer_info result_buffer{result.request()};
    my::math::Mat2x2 * tensors{reinterpret_cast<my::math::Mat2x2 *>(result_buffer.ptr)};

    cudaMemcpy(tensors, tensors_d, sizeof(my::math::Mat2x2) * N, cudaMemcpyDeviceToHost);
    MY_CUDA_ERROR_CHECK

    cudaFree(center_points_d);
    MY_CUDA_ERROR_CHECK
    cudaFree(tensors_d);
    MY_CUDA_ERROR_CHECK

    result.resize({static_cast<pybind11::ssize_t>(_cgproperties.DIM[0]), //
                   static_cast<pybind11::ssize_t>(_cgproperties.DIM[1]), //
                   static_cast<pybind11::ssize_t>(2),                    //
                   static_cast<pybind11::ssize_t>(2)});

    return result;
}