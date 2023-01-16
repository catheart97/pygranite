#include <cuda.h>

#include "granite/TrajectoryIntegrator.hpp" // only for ide

#ifdef __CUDACC__

/*
 * kernel part
 */

#ifdef __NVCC__

template <bool USE_SPHERE_COORDINATES>
__device__ __forceinline__ bool granite::out_of_box_x(const SimulationData<my::math::Vec3> & data,
                                                      my::math::Vec3 & v)
{
    return true; // never happens
}

template <bool USE_SPHERE_COORDINATES>
__device__ __forceinline__ bool granite::out_of_box_x(const SimulationData<my::math::Vec2> & data,
                                                      my::math::Vec2 & v)
{
    v.x = fmodf((v.x - data.Offset.X + data.Domain.X), data.Domain.X) + data.Offset.X;
    if (v.y - data.Offset.Y > data.Domain.Y || v.y - data.Offset.Y < 0) return true;
    return false;
}

template <bool USE_SPHERE_COORDINATES>
__device__ __forceinline__ bool granite::out_of_box_y(const SimulationData<my::math::Vec3> & data,
                                                      my::math::Vec3 & v)
{
    return true; // never happens
}

template <bool USE_SPHERE_COORDINATES>
__device__ __forceinline__ bool granite::out_of_box_y(const SimulationData<my::math::Vec2> & data,
                                                      my::math::Vec2 & v)
{
    if (USE_SPHERE_COORDINATES)
    {
        return out_of_box(data, v);
    }
    else
    {
        v.y = fmodf((v.y - data.Offset.Y + data.Domain.Y), data.Domain.Y) + data.Offset.Y;
        if (v.x - data.Offset.X > data.Domain.X || v.x - data.Offset.X < 0) return true;
        return false;
    }
}

template <bool USE_SPHERE_COORDINATES>
__device__ inline bool granite::out_of_box_xy(const SimulationData<my::math::Vec3> & data,
                                              const my::math::Vec3 &, my::math::Vec3 &)
{
    return true; // never happens
}

__device__ __forceinline__ float get_z(const my::math::Vec3 & v) { return v.z; }

__device__ __forceinline__ float get_z(const my::math::Vec2 & v) { return 0.f; }

template <bool USE_SPHERE_COORDINATES>
__device__ __forceinline__ bool granite::out_of_box_xy(const SimulationData<my::math::Vec2> & data,
                                                       const my::math::Vec2 & o, my::math::Vec2 & n)
{
    if (USE_SPHERE_COORDINATES)
    {
        my::math::Vec2 n_{(n.x - data.Offset.X), (n.y - data.Offset.Y)};

        if (n_.y < 0) // DOWN
        {
            my::math::Vec2 o_{(o.x - data.Offset.X), (o.y - data.Offset.Y)};

            auto tx{(-o_.y) / (n_.y - o_.y) * (n_.x - o_.x) + o_.x}; // intersection point
            n.y = -(n_.y) + data.Offset.Y;
            n.x = fmodf(data.Domain.X + 180.0f - n_.x + 2.f * tx, data.Domain.X) + data.Offset.X;
        }
        else if (n_.y >= data.Domain.Y) // UP
        {
            my::math::Vec2 o_{(o.x - data.Offset.X), (o.y - data.Offset.Y)};

            auto tx{(data.Domain.Y - o_.y) / (n_.y - o_.y) * (n_.x - o_.x) +
                    o_.x}; // intersection point
            n.y = (n_.y - data.Domain.Y) + data.Offset.Y;
            n.x = fmodf(data.Domain.X + 180.0f - n_.x + 2.f * tx, data.Domain.X) + data.Offset.X;
        }
    }
    else
    {
        n.y = fmodf((n.y - data.Offset.Y + data.Domain.Y), data.Domain.Y) + data.Offset.Y;
    }

    n.x = fmodf((n.x - data.Offset.X + data.Domain.X), data.Domain.X) + data.Offset.X;
    return false;
}

template <typename VEC_T>
__device__ __forceinline__ VEC_T granite::sphere_coordinates(SimulationData<VEC_T> & data, float py,
                                                             VEC_T & v)
{
    return VEC_T{(v.x * 180.f) / (data.Radius * cosf(py * my::math::PI_2 / 360.f) * my::math::PI),
                 (v.y * 180.f) / (my::math::PI * data.Radius)};
}

#endif

PYGRANITE_KERNEL_TEMPLATE_ARGS
__global__ void granite::integrate(SimulationData<VEC_T> data)
{
    using namespace my::math;

    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t NUM_THREADS = gridDim.x * blockDim.x;

    VEC_T * back = data.BackParticles;
    VEC_T * front = data.FrontParticles;

    for (size_t pid = idx; pid < data.NumParticles; pid += NUM_THREADS)
    {
        const VEC_T particle = back[pid];

        if (data.Status[pid] == AbortReason::Time)
        {
            VEC_T v, vc;
            float t{0.0f};
            if (UPLIFT_MODE == granite::UpLiftMode::Dynamic ||
                WINDFIELD_MODE == granite::WindfieldMode::Dynamic ||
                CONSTANTS_MODE == granite::ConstantsMode::Dynamic ||
                ADDITIONAL_VOLUME_MODE == granite::AdditionalVolumeMode::Dynamic)
            {
                t = my::math::unit_interval(data.Time, data.TimeFront, data.TimeBack);
            }

            if (WINDFIELD_MODE == granite::WindfieldMode::Dynamic)
            {
                switch (INTEGRATOR)
                {
                    case granite::Integrator::ClassicRungeKutta: {
                        const float & t0 = t;
                        const float t1 = my::math::unit_interval(data.Time + .5f * data.DeltaT,
                                                                 data.TimeFront, data.TimeBack);
                        const float t2 = my::math::unit_interval(data.Time + data.DeltaT,
                                                                 data.TimeFront, data.TimeBack);

                        if (USE_SPHERE_COORDINATES && SPACE == granite::Space::Space2D)
                        {
                            VEC_T k1c = interpolate(data, particle, t0);
                            const VEC_T k1 = sphere_coordinates(data, particle.y, k1c);

                            VEC_T p = particle + k1 * (0.5f * data.DeltaT);
                            VEC_T k2c = interpolate(data, p, t1);
                            const VEC_T k2 = sphere_coordinates(data, p.y, k2c);

                            p = particle + k2 * (0.5f * data.DeltaT);
                            VEC_T k3c = interpolate(data, p, t1);
                            const VEC_T k3 = sphere_coordinates(data, p.y, k3c);

                            p = particle + k3 * data.DeltaT;
                            VEC_T k4c = interpolate(data, p, t2);
                            const VEC_T k4 = sphere_coordinates(data, p.y, k4c);

                            if (COMP_ADDITIONAL) vc = (k1c + k2c * 2 + k3c * 2 + k4c) * (1.0f / 6);
                            v = (k1 + k2 * 2 + k3 * 2 + k4) * (1.0f / 6);
                        }
                        else
                        {
                            const VEC_T k1 = interpolate(data, particle, t0);
                            const VEC_T k2 =
                                interpolate(data, particle + k1 * (0.5f * data.DeltaT), t1);
                            const VEC_T k3 =
                                interpolate(data, particle + k2 * (0.5f * data.DeltaT), t1);
                            const VEC_T k4 = interpolate(data, particle + k3 * data.DeltaT, t2);

                            v = (k1 + k2 * 2 + k3 * 2 + k4) * (1.0f / 6);
                            vc = v;
                        }
                    }
                    break;
                    default: {
                        vc = interpolate(data, particle, t);
                        v = vc;
                        if (USE_SPHERE_COORDINATES && SPACE == granite::Space::Space2D)
                            v = sphere_coordinates(data, particle.y, v);
                    }
                    break;
                }
            }
            else
            {
                switch (INTEGRATOR)
                {
                    case granite::Integrator::ClassicRungeKutta: {
                        if (USE_SPHERE_COORDINATES && SPACE == granite::Space::Space2D)
                        {
                            VEC_T k1c = interpolate(data, particle);
                            const VEC_T k1 = sphere_coordinates(data, particle.y, k1c);
                            
                            VEC_T p = particle + k1 * (0.5f * data.DeltaT);
                            VEC_T k2c = interpolate(data, p);
                            const VEC_T k2 = sphere_coordinates(data, p.y, k2c);
                            p = particle + k2 * (0.5f * data.DeltaT);
                            VEC_T k3c = interpolate(data, p);
                            const VEC_T k3 = sphere_coordinates(data, p.y, k3c);
                            p = particle + k3 * data.DeltaT;
                            VEC_T k4c = interpolate(data, p);
                            const VEC_T k4 = sphere_coordinates(data, p.y, k4c);

                            if (COMP_ADDITIONAL) vc = (k1c + k2c * 2 + k3c * 2 + k4c) * (1.0f / 6);
                            v = (k1 + k2 * 2 + k3 * 2 + k4) * (1.0f / 6);
                        }
                        else
                        {
                            VEC_T k1 = interpolate(data, particle);
                            VEC_T k2 = interpolate(data, particle + k1 * (0.5f * data.DeltaT));
                            VEC_T k3 = interpolate(data, particle + k2 * (0.5f * data.DeltaT));
                            VEC_T k4 = interpolate(data, particle + k3 * data.DeltaT);

                            v = (k1 + k2 * 2 + k3 * 2 + k4) * (1.0f / 6);
                            vc = v;
                        }
                    }
                    break;
                    default: {
                        v = interpolate(data, particle);
                        vc = v;
                        if (USE_SPHERE_COORDINATES && SPACE == granite::Space::Space2D)
                            v = sphere_coordinates(data, particle.y, v);
                    }
                }
            }

            // UpLift is always positive (even on reverse computation)
            VEC_T dp;
            if (UPLIFT_MODE != granite::UpLiftMode::Off)
            {
                dp = v * data.DeltaT + my::math::lerp(data.LiftFront[pid], data.LiftBack[pid], t) *
                                           (USE_REVERSE_COMPUTATION ? -1 : 1);
            }
            else
            {
                dp = v * data.DeltaT;
            }

            AbortReason abort{AbortReason::Time};
            if (ABORT_MODE != granite::AbortMode::Time) // FitLength or Length
            {
                const float dp_l{length(dp)};
                const float c_l{data.CurrentLength[pid]};
                const float n_l{c_l + dp_l};

                if (ABORT_MODE == granite::AbortMode::FitLength)
                {
                    if (n_l > data.MaximumLength)
                    {
                        dp = ((data.MaximumLength - c_l) / dp_l) * dp;
                        data.CurrentLength[pid] = data.MaximumLength;
                        abort = AbortReason::Length;
                    }
                    else
                    {
                        data.CurrentLength[pid] = n_l;
                    }
                }
                else
                {
                    if (n_l > data.MaximumLength)
                    {
                        abort = AbortReason::Length;
                    }
                    data.CurrentLength[pid] = n_l;
                }
            }

            VEC_T new_pos = USE_REVERSE_COMPUTATION ? (particle - dp) : (particle + dp);

            if (abort == AbortReason::Time)
            {
                switch (BORDER_MODE)
                {
                    case granite::BorderMode::LoopX: {
                        if (length_squared(dp) < granite::ABORT_VALUE)
                            abort = AbortReason::Wind;
                        else if (out_of_box_x<USE_SPHERE_COORDINATES>(data, new_pos))
                            abort = AbortReason::Domain;
                        break;
                    }
                    case granite::BorderMode::LoopY: {
                        if (length_squared(dp) < granite::ABORT_VALUE)
                            abort = AbortReason::Wind;
                        else if (out_of_box_y<USE_SPHERE_COORDINATES>(data, new_pos))
                            abort = AbortReason::Domain;
                        break;
                    }
                    case granite::BorderMode::LoopXY: {
                        if (length_squared(dp) < granite::ABORT_VALUE)
                            abort = AbortReason::Wind;
                        else if (out_of_box_xy<USE_SPHERE_COORDINATES>(data, particle, new_pos))
                            abort = AbortReason::Domain;
                        break;
                    }
                    default: {
                        if (USE_TOPOGRAPHY && in_topography(data, new_pos))
                        {
                            abort = AbortReason::Topography;
                            break;
                        }

                        if (length_squared(dp) < granite::ABORT_VALUE)
                            abort = AbortReason::Wind;
                        else if (out_of_box(data, new_pos))
                            abort = AbortReason::Domain;
                        break;
                    }
                }
            }

            data.Status[pid] = abort;

            if (abort == AbortReason::Wind || abort == AbortReason::Topography ||
                abort == AbortReason::Domain)
            {
                front[pid] = particle;
                if (CURVATURE_MODE != CurvatureMode::Off) data.TotalCurvature[pid] = -1.f;
            }
            else
            {
                if (CURVATURE_MODE != CurvatureMode::Off)
                {
                    if (CURVATURE_MODE == CurvatureMode::FastTotalCurvature)
                    {
                        const VEC_T v_ = normalized(v);
                        if (data.Time > 0)
                        {
                            auto k{acos(dot(data.LastDirections[pid], v_))};
                            if (!isnan(k)) data.TotalCurvature[pid] += k;
                        }
                        data.LastDirections[pid] = v_;
                    }
                    else if (data.Time > 0)
                    {
                        const VEC_T pm1{front[pid]}; // p(t - dt)
                        const VEC_T & p{particle};   // p(t)
                        const VEC_T & pp1{new_pos};  // p(t + dt)

                        auto k{abs(granite::kappa(pm1, p, pp1, data.DeltaT))};
                        auto dt{.5f * (length(p - pm1) + length(pp1 - p))};

                        if (CURVATURE_MODE == CurvatureMode::IndividualAndTotalCurvature)
                            data.Curvature[pid] = k;

                        if (!isnan(k)) data.TotalCurvature[pid] += k * dt;
                    }
                }

                if (ADDITIONAL_VOLUME_MODE != granite::AdditionalVolumeMode::Off)
                {
                    VEC_T arg{texture_arg(data, new_pos)};
                    for (size_t i = 0; i < data.NumAdditionalVolumes; ++i)
                    {
                        if (ADDITIONAL_VOLUME_MODE == granite::AdditionalVolumeMode::Constant)
                            data.ParticleVolumeInfo[i][pid] = interpolate_volume(data, i, arg);
                        else
                        {
                            data.ParticleVolumeInfo[i][pid] =
                                interpolate_volume(data, data.AdditionalVolumesBack[i],
                                                   data.AdditionalVolumesFront[i], arg, t);
                        }
                    }
                }

                if (COMP_ADDITIONAL)
                {
                    const float ref[]{
                        new_pos.x,      //
                        new_pos.y,      //
                        get_z(new_pos), // function wrapper for z value as 2D and 3D sharing code
                        CURVATURE_MODE == CurvatureMode::IndividualAndTotalCurvature
                            ? data.Curvature[pid]
                            : 0.f,                                                             //
                        data.NumAdditionalVolumes > 0 ? data.ParticleVolumeInfo[0][pid] : 0.f, //
                        data.NumAdditionalVolumes > 1 ? data.ParticleVolumeInfo[1][pid] : 0.f, //
                        data.NumAdditionalVolumes > 2 ? data.ParticleVolumeInfo[2][pid] : 0.f, //
                        data.NumAdditionalVolumes > 3 ? data.ParticleVolumeInfo[3][pid] : 0.f, //
                        data.NumAdditionalVolumes > 4 ? data.ParticleVolumeInfo[4][pid] : 0.f, //
                        data.NumAdditionalVolumes > 5 ? data.ParticleVolumeInfo[5][pid] : 0.f, //
                        data.NumAdditionalVolumes > 6 ? data.ParticleVolumeInfo[6][pid] : 0.f, //
                        data.NumAdditionalVolumes > 7 ? data.ParticleVolumeInfo[7][pid] : 0.f, //
                        data.NumAdditionalConstants > 0                                        //
                            ? (data.ComputeDataBack[0]                                         //
                                   ? (CONSTANTS_MODE == granite::ConstantsMode::Dynamic        //
                                          ? my::math::lerp(data.ComputeDataBack[0][pid],       //
                                                           data.ComputeDataFront[0][pid],      //
                                                           t)                                  //
                                          : data.ComputeDataBack[0][pid]                       //
                                      )                                                        //
                                   : 0.f)                                                      //
                            : 0.f,                                                             //
                        data.NumAdditionalConstants > 1                                        //
                            ? (data.ComputeDataBack[1]                                         //
                                   ? (CONSTANTS_MODE == granite::ConstantsMode::Dynamic        //
                                          ? my::math::lerp(data.ComputeDataBack[1][pid],       //
                                                           data.ComputeDataFront[1][pid],      //
                                                           t)                                  //
                                          : data.ComputeDataBack[1][pid]                       //
                                      )                                                        //
                                   : 0.f)                                                      //
                            : 0.f,                                                             //
                        data.NumAdditionalConstants > 2                                        //
                            ? (data.ComputeDataBack[2]                                         //
                                   ? (CONSTANTS_MODE == granite::ConstantsMode::Dynamic        //
                                          ? my::math::lerp(data.ComputeDataBack[2][pid],       //
                                                           data.ComputeDataFront[2][pid],      //
                                                           t)                                  //
                                          : data.ComputeDataBack[2][pid]                       //
                                      )                                                        //
                                   : 0.f)                                                      //
                            : 0.f,                                                             //
                        data.NumAdditionalConstants > 3                                        //
                            ? (data.ComputeDataBack[3]                                         //
                                   ? (CONSTANTS_MODE == granite::ConstantsMode::Dynamic        //
                                          ? my::math::lerp(data.ComputeDataBack[3][pid],       //
                                                           data.ComputeDataFront[3][pid],      //
                                                           t)                                  //
                                          : data.ComputeDataBack[3][pid]                       //
                                      )                                                        //
                                   : 0.f)                                                      //
                            : 0.f,                                                             //
                        vc.x,                                                                  //
                        vc.y,                                                                  //
                        get_z(vc)                                                              //
                    };

                    for (auto i = 0; i < data.NumAdditionalCompute; ++i)
                    {
                        data.ParticleComputeInfo[i][pid] =
                            interpret<PYGRANITE_KERNEL_TEMPLATE_CALL>(data.ASTNodes[i],
                                                                      data.ASTSizes[i], ref, 0);
                    }
                }

                front[pid] = new_pos;
            }
        }
        else
        {
            front[pid] = particle;
        }
    }
}

#endif

/*
 * host part
 */

#ifdef __NVCC__

PYGRANITE_KERNEL_TEMPLATE_ARGS
void granite::TrajectoryIntegrator::compute_(std::vector<std::vector<VEC_T>> & traj)
{
    if (traj.size() == 0) MY_USER_ERROR("Provided TrajectorySet does not contain any data.");

    //// ! Setup device particles
    MY_VLOG("SETTING UP DEVICE PARTICLE BUFFERS ...");
    std::vector<VEC_T> & start_particles_vector = traj[traj.size() - 1];
    VEC_T * start_particles_host = start_particles_vector.data();

    if (_particles_device.front()) cudaFree(_particles_device.front());
    if (_particles_device.back()) cudaFree(_particles_device.back());
    _particles_device.front() = nullptr;
    _particles_device.back() = nullptr;
    _particles_device.reset();

    VEC_T * start_particles_device[2]{nullptr, nullptr};

    cudaMalloc(&(start_particles_device[0]), start_particles_vector.size() * sizeof(VEC_T));
    MY_CUDA_ERROR_CHECK

    cudaMalloc(&(start_particles_device[1]), start_particles_vector.size() * sizeof(VEC_T));
    MY_CUDA_ERROR_CHECK

    // initialize particles in front, as it will be swapped before a computation
    cudaMemcpy(start_particles_device[0],                     //
               start_particles_host,                          //
               start_particles_vector.size() * sizeof(VEC_T), //
               cudaMemcpyHostToDevice);
    MY_CUDA_ERROR_CHECK
    MY_VLOG("SETTING UP DEVICE PARTICLE BUFFERS ... DONE");
    //// !

    //// ! Setup device particle status
    MY_VLOG("SETTING UP DEVICE PARTICLE STATUS BUFFER ...");
    AbortReason * particles_status_device{nullptr}; // tracks whether dead or not
    cudaMalloc(&particles_status_device, start_particles_vector.size() * sizeof(AbortReason));
    MY_CUDA_ERROR_CHECK

    cudaMemset(particles_status_device, 0, start_particles_vector.size() * sizeof(AbortReason));
    MY_CUDA_ERROR_CHECK
    MY_VLOG("SETTING UP DEVICE PARTICLE STATUS BUFFER ... DONE");
    //// !

    _particles_device =
        my::util::SwapChain<float *>(reinterpret_cast<float *>(start_particles_device[0]),
                                     reinterpret_cast<float *>(start_particles_device[1]));

    //// ! Simulation data object
    SimulationData<VEC_T> data;
    data.Status = particles_status_device;
    data.NumParticles = start_particles_vector.size();
    data.Radius = _settings.SphereRadius;

    if (_settings.GridScale.size() == 1)
    {
        data.GridScale.X = _settings.GridScale[0];
        data.GridScale.Y = _settings.GridScale[0];
        data.GridScale.Z = _settings.GridScale[0];
        data.Domain.X = _grid.X * _settings.GridScale[0];
        data.Domain.Y = _grid.Y * _settings.GridScale[0];
        data.Domain.Z = _grid.Z * _settings.GridScale[0];
    }
    else if (_settings.GridScale.size() == 2)
    {
        data.GridScale.X = _settings.GridScale[0];
        data.GridScale.Y = _settings.GridScale[1];
        data.GridScale.Z = _settings.GridScale[0];
        data.Domain.X = _grid.X * _settings.GridScale[0];
        data.Domain.Y = _grid.Y * _settings.GridScale[1];
        data.Domain.Z = _grid.Z * _settings.GridScale[0];
    }
    else if (_settings.GridScale.size() == 3)
    {
        data.GridScale.X = _settings.GridScale[0];
        data.GridScale.Y = _settings.GridScale[1];
        data.GridScale.Z = _settings.GridScale[2];
        data.Domain.X = _grid.X * _settings.GridScale[0];
        data.Domain.Y = _grid.Y * _settings.GridScale[1];
        data.Domain.Z = _grid.Z * _settings.GridScale[2];
    }
    else
    {
        MY_USER_ERROR("Invalid GridScale dimension.");
    }

    if (_settings.Offset.size() == 1)
    {
        data.Offset.X = _settings.Offset[0];
        data.Offset.Y = _settings.Offset[0];
        data.Offset.Z = _settings.Offset[0];
    }
    else if (_settings.GridScale.size() == 2)
    {
        data.Offset.X = _settings.Offset[0];
        data.Offset.Y = _settings.Offset[1];
        data.Offset.Z = _settings.Offset[0];
    }
    else if (_settings.GridScale.size() == 3)
    {
        data.Offset.X = _settings.Offset[0];
        data.Offset.Y = _settings.Offset[1];
        data.Offset.Z = _settings.Offset[2];
    }
    else
    {
        MY_USER_ERROR("Invalid Offset dimension.");
    }
    //// !

    //// ! Initialize additional abort data
    if (ABORT_MODE != AbortMode::Time)
    {
        float * length_device{nullptr};
        cudaMalloc(&length_device, start_particles_vector.size() * sizeof(float));
        MY_CUDA_ERROR_CHECK

        cudaMemset(length_device, 0, start_particles_vector.size() * sizeof(float));
        MY_CUDA_ERROR_CHECK

        data.MaximumLength = _settings.MaximumLength;
        data.CurrentLength = length_device;
    }
    //// !

    //// ! Initialize curvature
    if (CURVATURE_MODE != CurvatureMode::Off)
    {
        MY_VLOG("SETTING UP DEVICE CURVATURE BUFFERS ...");
        if (CURVATURE_MODE == CurvatureMode::IndividualAndTotalCurvature)
        {
            float * curvature_device{nullptr};
            cudaMalloc(&curvature_device, start_particles_vector.size() * sizeof(float));
            MY_CUDA_ERROR_CHECK

            cudaMemset(curvature_device, 0, start_particles_vector.size() * sizeof(float));
            MY_CUDA_ERROR_CHECK

            data.Curvature = curvature_device;
        }

        float * total_curvature_device{nullptr};
        VEC_T * last_direction_device{nullptr};
        if (CURVATURE_MODE == CurvatureMode::FastTotalCurvature)
        {
            cudaMalloc(&last_direction_device, start_particles_vector.size() * sizeof(VEC_T));
            MY_CUDA_ERROR_CHECK
        }
        cudaMalloc(&total_curvature_device, start_particles_vector.size() * sizeof(float));
        MY_CUDA_ERROR_CHECK

        cudaMemset(total_curvature_device, 0, start_particles_vector.size() * sizeof(float));
        MY_CUDA_ERROR_CHECK

        data.TotalCurvature = total_curvature_device;
        data.LastDirections = last_direction_device;
        MY_VLOG("SETTING UP DEVICE CURVATURE BUFFERS ... DONE");
    }

    //// ! Additional Volume
    if (ADDITIONAL_VOLUME_MODE != granite::AdditionalVolumeMode::Off)
    {
        MY_VLOG("SETTING UP DEVICE VOLUME BUFFERS ...");
        data.NumAdditionalVolumes = _num_additional_volumes;

        for (size_t i = 0; i < data.NumAdditionalVolumes; ++i)
        {
            float * particle_volume_info_device{nullptr};
            cudaMalloc(&particle_volume_info_device, start_particles_vector.size() * sizeof(float));
            MY_CUDA_ERROR_CHECK

            data.ParticleVolumeInfo[i] = particle_volume_info_device;
        }
        MY_VLOG("SETTING UP DEVICE VOLUME BUFFERS ... DONE");
    }
    //// !

    //// ! Initialize additional compute data
    MY_VLOG("SETTING UP ADDITIONAL COMPUTE ...")
    
    if (COMP_ADDITIONAL)
    {
        // setup interpreter data
        _set->_num_additional_compute = _settings.AdditionalCompute.size();
        data.NumAdditionalCompute = _set->_num_additional_compute;

        for (size_t i = 0; i < std::min(_settings.AdditionalCompute.size(), MAX_ADDITIONAL_COMPUTE);
             ++i)
        {
            std::string code = _settings.AdditionalCompute[i];

            auto ast = granite::parse(code);
            ast = replace_environment(ast, _env);

            if (SPACE == granite::Space::Space2D &&
                contains(ast, granite::ASTNodeType::Reference_Z))
                MY_USER_ERROR("Compute code contains undefinded token 'z'");
            if (SPACE == granite::Space::Space2D &&
                contains(ast, granite::ASTNodeType::Reference_W))
                MY_USER_ERROR("Compute code contains undefinded token 'w'");

            if (data.NumAdditionalVolumes < 8 && contains(ast, granite::ASTNodeType::Reference_F7))
                MY_USER_ERROR("Compute code contains undefinded token 'f7'");
            if (data.NumAdditionalVolumes < 7 && contains(ast, granite::ASTNodeType::Reference_F6))
                MY_USER_ERROR("Compute code contains undefinded token 'f6'");
            if (data.NumAdditionalVolumes < 6 && contains(ast, granite::ASTNodeType::Reference_F5))
                MY_USER_ERROR("Compute code contains undefinded token 'f5'");
            if (data.NumAdditionalVolumes < 5 && contains(ast, granite::ASTNodeType::Reference_F4))
                MY_USER_ERROR("Compute code contains undefinded token 'f4'");
            if (data.NumAdditionalVolumes < 4 && contains(ast, granite::ASTNodeType::Reference_F3))
                MY_USER_ERROR("Compute code contains undefinded token 'f3'");
            if (data.NumAdditionalVolumes < 3 && contains(ast, granite::ASTNodeType::Reference_F2))
                MY_USER_ERROR("Compute code contains undefinded token 'f2'");
            if (data.NumAdditionalVolumes < 2 && contains(ast, granite::ASTNodeType::Reference_F1))
                MY_USER_ERROR("Compute code contains undefinded token 'f1'");
            if (data.NumAdditionalVolumes < 1 && contains(ast, granite::ASTNodeType::Reference_F0))
                MY_USER_ERROR("Compute code contains undefinded token 'f0'");

            if (_num_additional_constants < 1 && contains(ast, granite::ASTNodeType::Reference_C0))
                MY_USER_ERROR("Compute code contains undefinded token 'c0'");
            if (_num_additional_constants < 2 && contains(ast, granite::ASTNodeType::Reference_C1))
                MY_USER_ERROR("Compute code contains undefinded token 'c1'");
            if (_num_additional_constants < 3 && contains(ast, granite::ASTNodeType::Reference_C2))
                MY_USER_ERROR("Compute code contains undefinded token 'c2'");
            if (_num_additional_constants < 4 && contains(ast, granite::ASTNodeType::Reference_C3))
                MY_USER_ERROR("Compute code contains undefinded token 'c3'");

            if (CURVATURE_MODE != granite::CurvatureMode::IndividualAndTotalCurvature &&
                contains(ast, granite::ASTNodeType::Reference_CV))
                MY_USER_ERROR("Compute code contains undefinded token 'cv'");

            auto gpu_program = granite::flatten(ast);
            cudaMalloc(&data.ASTNodes[i], sizeof(granite::ASTANode) * gpu_program.size());
            MY_CUDA_ERROR_CHECK
            cudaMemcpy(data.ASTNodes[i], gpu_program.data(),
                       sizeof(granite::ASTANode) * gpu_program.size(), cudaMemcpyHostToDevice);
            MY_CUDA_ERROR_CHECK

            data.ASTSizes[i] = gpu_program.size();

            cudaMalloc(&data.ParticleComputeInfo[i], sizeof(float) * data.NumParticles);
            MY_CUDA_ERROR_CHECK
            cudaMemset(data.ParticleComputeInfo[i], 0, sizeof(float) * data.NumParticles);
            MY_CUDA_ERROR_CHECK
        }
    }
    MY_VLOG("SETTING UP ADDITIONAL COMPUTE ... DONE")

    computeLoop_<PYGRANITE_KERNEL_TEMPLATE_CALL>(data);

    MY_VLOG("FREEING DEVICE BUFFERS ...");

    //// ! Cleanup status
    {
        _set->_abort_reasons.resize(data.NumParticles);
        cudaMemcpy(_set->_abort_reasons.data(), data.Status,
                   sizeof(AbortReason) * data.NumParticles, cudaMemcpyDeviceToHost);
        MY_CUDA_ERROR_CHECK
        cudaFree(particles_status_device);
    }

    //// !

    //// ! Cleanup abort
    if (ABORT_MODE != AbortMode::Time)
    {
        _set->_individual_lengths.resize(data.NumParticles);
        cudaMemcpy(_set->_individual_lengths.data(), data.CurrentLength,
                   sizeof(float) * data.NumParticles, cudaMemcpyDeviceToHost);
        MY_CUDA_ERROR_CHECK
        cudaFree(data.CurrentLength);
    }
    //// !

    //// ! Cleanup curvature
    if (CURVATURE_MODE != CurvatureMode::Off)
    {
        if (CURVATURE_MODE == CurvatureMode::IndividualAndTotalCurvature) cudaFree(data.Curvature);

        _set->_total_curvature.resize(data.NumParticles);
        cudaMemcpy(_set->_total_curvature.data(), data.TotalCurvature,
                   sizeof(float) * data.NumParticles, cudaMemcpyDeviceToHost);
        MY_CUDA_ERROR_CHECK
        cudaFree(data.TotalCurvature);

        if (CURVATURE_MODE == CurvatureMode::FastTotalCurvature) cudaFree(data.LastDirections);
    }
    //// !

    if (COMP_ADDITIONAL)
    {
        for (size_t i = 0; i < std::min(_settings.AdditionalCompute.size(), MAX_ADDITIONAL_COMPUTE);
             ++i)
        {
            cudaFree(data.ASTNodes[i]);
            cudaFree(data.ParticleComputeInfo[i]);
        }
    }

    //// ! Cleanup additional volume
    if (COMP_ADDITIONAL_VOLUME)
    {
        for (size_t i = 0; i < data.NumAdditionalVolumes; ++i)
        {
            cudaFree(data.ParticleVolumeInfo[i]);
        }
    }
    //// !
    MY_VLOG("FREEING DEVICE BUFFERS ... DONE");
}

PYGRANITE_KERNEL_TEMPLATE_ARGS
void granite::TrajectoryIntegrator::computeLoop_(SimulationData<VEC_T> & data)
{
    data.Time = 0;
    size_t counter = 0;
    float max_windfield_time{_settings.DataTimeDistance};

    MY_VLOG("COMPUTING OPTIMAL KERNEL PARAMETERS ...");

    // compute optimal amount of gpu threads
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    const unsigned int THREADS =
        static_cast<unsigned int>(
            std::min(static_cast<size_t>(devProp.maxThreadsPerBlock), data.NumParticles)) /
        2;
    const unsigned int BLOCKS =
        static_cast<unsigned int>(std::min(my::math::sdiv<size_t>(THREADS, data.NumParticles),
                                           static_cast<size_t>(devProp.maxThreadsDim[0])));

    int32_t * number_alive_device;
    cudaMalloc(&number_alive_device, sizeof(int32_t));

    // checks if at least one particle is alive (false = alive, true = dead)
    bool abort_due_alive{false};
    std::function<bool(void)> alive = [&]() {
        int32_t number_alive{0};
        my::cuda::count_if<int32_t, int32_t><<<BLOCKS, THREADS>>>(
            reinterpret_cast<int32_t *>(data.Status), number_alive_device, 0, data.NumParticles);
        cudaMemcpy(&number_alive, number_alive_device, sizeof(int32_t), cudaMemcpyDeviceToHost);
        bool tmp = (number_alive >= _settings.MinimumAliveParticles);
        abort_due_alive = !tmp;
        return tmp;
    };

    MY_VLOG("COMPUTING OPTIMAL KERNEL PARAMETERS ... DONE");

    MY_VLOG("SETTING UP CONSTANT DATA ... ");
    if (WINDFIELD_MODE == granite::WindfieldMode::Constant)
    {
        data.WindfieldBack[0] = _windfield_textures[0][0].Object;
        data.WindfieldBack[1] = _windfield_textures[0][1].Object;
        data.WindfieldBack[2] = _windfield_textures[0][2].Object;
        // data.TimeBack = COUNTER * _settings.DataTimeDistance;
    }

    if (ADDITIONAL_VOLUME_MODE == granite::AdditionalVolumeMode::Constant)
    {
        for (size_t i = 0; i < data.NumAdditionalVolumes; ++i)
        {
            data.AdditionalVolumesBack[i] = _additional_textures[_back_index][i].Object;
        }
    }

    if (UPLIFT_MODE == granite::UpLiftMode::Constant)
    {
        data.LiftBack = reinterpret_cast<VEC_T *>(_uplift_device[0]);
    }

    data.NumAdditionalConstants = _num_additional_constants;
    if (CONSTANTS_MODE == granite::ConstantsMode::Constant)
    {
        for (size_t i = 0; i < _num_additional_constants; ++i)
        {
            data.ComputeDataBack[i] = _constants_device[0][i];
            data.ComputeDataFront[i] = _constants_device[0][i];
        }
    }

    data.DeltaT = _settings.DeltaT;
    MY_VLOG("SETTING UP CONSTANT DATA ... DONE");

    MY_VLOG("LAUNCHING COMPUTE LOOP")

    do
    {
        MY_VLOG("INITIATING COMPUTE THREAD")

        std::thread compute_gpu;
        if (_settings.WindfieldMode == granite::WindfieldMode::Dynamic ||
            _settings.UpLiftMode == granite::UpLiftMode::Dynamic ||
            _settings.AdditionalVolumeMode == granite::AdditionalVolumeMode::Dynamic ||
            _settings.ConstantsMode == granite::ConstantsMode::Dynamic)
        {
            computeLoopGPU_<PYGRANITE_KERNEL_TEMPLATE_CALL>(data,               //
                                                            alive,              //
                                                            max_windfield_time, //
                                                            counter,            //
                                                            BLOCKS,             //
                                                            THREADS);
            compute_gpu = std::thread();
        }
        else
        {
            compute_gpu = std::thread([&]() {
                computeLoopGPU_<PYGRANITE_KERNEL_TEMPLATE_CALL>(data,                            //
                                                                alive,                           //
                                                                _settings.MaximumSimulationTime, //
                                                                counter,                         //
                                                                BLOCKS,                          //
                                                                THREADS);
            });
        }

        if (_settings.WindfieldMode == granite::WindfieldMode::Dynamic ||
            _settings.UpLiftMode == granite::UpLiftMode::Dynamic ||
            _settings.AdditionalVolumeMode == granite::AdditionalVolumeMode::Dynamic ||
            _settings.ConstantsMode == granite::ConstantsMode::Dynamic)
        {
            if (_loader.step())
            {
                updateData(_cache_index);

                if (abort_due_alive) break;
                max_windfield_time += _settings.DataTimeDistance;

                // swap texture indices
                MY_VLOG("SWAPPING TEXTURE INDICES")
                size_t tmp = _back_index;
                _back_index = _front_index;
                _front_index = _cache_index;
                _cache_index = tmp;

                // increase windfield counter
                counter++;
            }
            else
            {
                if (compute_gpu.joinable()) compute_gpu.join();
                break;
            }
        }
        if (compute_gpu.joinable()) compute_gpu.join();
    } while (WINDFIELD_MODE == granite::WindfieldMode::Dynamic &&
             _simulation_counter < (_settings.MaximumSimulationTime / _settings.DeltaT));

    if (_settings.SaveInterval == 0)
        copyParticleInfo<VEC_T, CURVATURE_MODE, COMP_ADDITIONAL, COMP_ADDITIONAL_VOLUME>(data);

    cudaFree(number_alive_device);
}

PYGRANITE_KERNEL_TEMPLATE_ARGS
void granite::TrajectoryIntegrator::computeLoopGPU_(SimulationData<VEC_T> & data,
                                                    const std::function<bool(void)> & alive, //
                                                    const float MAXIMUM_WINDFIELD_TIME,      //
                                                    const size_t COUNTER,                    //
                                                    const unsigned int BLOCKS,               //
                                                    const unsigned int THREADS)
{
    if (USE_TOPOGRAPHY)
    {
        data.Topography = _topography_texture.Object;
    }

    data.TimeBack = COUNTER * _settings.DataTimeDistance;
    data.TimeFront = (COUNTER + 1) * _settings.DataTimeDistance;

    if (WINDFIELD_MODE == granite::WindfieldMode::Dynamic)
    {
        data.WindfieldBack[0] = _windfield_textures[_back_index][0].Object;
        data.WindfieldBack[1] = _windfield_textures[_back_index][1].Object;
        data.WindfieldBack[2] = _windfield_textures[_back_index][2].Object;

        data.WindfieldFront[0] = _windfield_textures[_front_index][0].Object;
        data.WindfieldFront[1] = _windfield_textures[_front_index][1].Object;
        data.WindfieldFront[2] = _windfield_textures[_front_index][2].Object;
    }

    if (ADDITIONAL_VOLUME_MODE == granite::AdditionalVolumeMode::Dynamic)
    {
        for (size_t i = 0; i < data.NumAdditionalVolumes; ++i)
        {
            data.AdditionalVolumesBack[i] = _additional_textures[_back_index][i].Object;
            data.AdditionalVolumesFront[i] = _additional_textures[_front_index][i].Object;
        }
    }

    if (UPLIFT_MODE == granite::UpLiftMode::Dynamic)
    {
        data.LiftBack = reinterpret_cast<VEC_T *>(_uplift_device[_back_index]);
        data.LiftFront = reinterpret_cast<VEC_T *>(_uplift_device[_front_index]);
    }

    if (CONSTANTS_MODE == granite::ConstantsMode::Dynamic)
    {
        for (size_t i = 0; i < _num_additional_constants; ++i)
        {
            data.ComputeDataBack[i] = _constants_device[_back_index][i];
            data.ComputeDataFront[i] = _constants_device[_front_index][i];
        }
    }

    MY_VLOG("Initializing computation loop.")

    while ( //
        _simulation_counter < (_settings.MaximumSimulationTime / _settings.DeltaT) &&
        (!_settings.MinimumAliveParticles || alive()))
    {

        _particles_device.swap();
        data.FrontParticles = reinterpret_cast<VEC_T *>(_particles_device.front());
        data.BackParticles = reinterpret_cast<VEC_T *>(_particles_device.back());

        integrate<PYGRANITE_KERNEL_TEMPLATE_CALL><<<BLOCKS, THREADS>>>(data);

        cudaDeviceSynchronize();
        MY_CUDA_ERROR_CHECK

        if (_settings.SaveInterval != 0 && (_simulation_counter) % _settings.SaveInterval == 0)
            copyParticleInfo<VEC_T, CURVATURE_MODE, COMP_ADDITIONAL, COMP_ADDITIONAL_VOLUME>(data);

        data.Time += data.DeltaT;
        _simulation_counter++;

        if (data.Time >= MAXIMUM_WINDFIELD_TIME) break;
    }
}

template <typename VEC_T, granite::CurvatureMode CURVATURE_MODE, bool COMP_ADDITIONAL,
          bool COMP_ADDITIONAL_VOLUME>
void granite::TrajectoryIntegrator::copyParticleInfo(const SimulationData<VEC_T> & data)
{
    copyParticlePositions(data.NumParticles, //
                          reinterpret_cast<VEC_T *>(_particles_device.front()));

    if (CURVATURE_MODE == CurvatureMode::IndividualAndTotalCurvature && data.Time > 0)
    {
        std::vector<float> step_data(data.NumParticles);
        cudaMemcpy(step_data.data(), data.Curvature, data.NumParticles * sizeof(float),
                   cudaMemcpyDeviceToHost);
        MY_CUDA_ERROR_CHECK
        step_data.shrink_to_fit();
        _set->_curvature.push_back(step_data);
    }

    if (COMP_ADDITIONAL)
    {
        auto num{data.NumAdditionalCompute};

        for (size_t i = 0; i < num; ++i)
        {
            std::vector<float> step_data(data.NumParticles);
            cudaMemcpy(step_data.data(), data.ParticleComputeInfo[i],
                       data.NumParticles * sizeof(float), cudaMemcpyDeviceToHost);
            MY_CUDA_ERROR_CHECK
            step_data.shrink_to_fit();
            _set->_compute_data[i].push_back(step_data);
        }

        _set->_num_additional_volumes = num;
    }

    if (COMP_ADDITIONAL_VOLUME)
    {
        auto num{data.NumAdditionalVolumes};

        for (size_t i = 0; i < _keys.size(); ++i)
        {
            std::vector<float> step_data(data.NumParticles);
            cudaMemcpy(step_data.data(), data.ParticleVolumeInfo[i],
                       data.NumParticles * sizeof(float), cudaMemcpyDeviceToHost);
            MY_CUDA_ERROR_CHECK
            step_data.shrink_to_fit();
            _set->_volume_data[_keys[i]].push_back(step_data);
            i++;
            if (i >= num) break;
        }

        _set->_num_additional_volumes = num;
    }
}

#endif
