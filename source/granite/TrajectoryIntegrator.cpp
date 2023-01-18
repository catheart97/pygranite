#include "granite/TrajectoryIntegrator.hpp"
#include "../../third_party/cpp-peglib/peglib.h"

std::shared_ptr<granite::ASTTNode> granite::parse(const std::string & code)
{
    using namespace granite;

    peg::parser parser;

    std::string_view grammar = R"(
        Expression <- OperatorExpression(PrimaryExpression, Operator)
        PrimaryExpression <- '(' Expression ')' / FunctionExpression / Register / OldRegister / Identifier / Constant
        Constant <- < [-]?[0-9]+[.][0-9]+ >  /  < [-]?[0-9]+ >
        Operator <-  < [-+/*] >
        FunctionExpression <- 'sqrt' '(' Expression ')' / 'exp' '(' Expression ')' / 'ln' '(' Expression ')' / 'sin' '(' Expression ')' / 'cos' '(' Expression ')' / 'tan' '(' Expression ')' / 'asin' '(' Expression ')' / 'acos' '(' Expression ')' / 'atan' '(' Expression ')' / 'atan2' '(' Expression ',' Expression ')'
        Register <- 'x' | 'y' | 'z' | 'cv' | 'u' | 'v' | 'w'
        OldRegister <- 'f0' | 'f1' | 'f2' | 'f3' | 'f4' | 'f5' | 'f6' | 'f7' | 'c0' | 'c1' | 'c2' | 'c3'
        %whitespace <- [ 	]*
        Identifier <- < [_a-zA-Z][a-zA-Z0-9_]* >

        OperatorExpression(A, O) <-  A (O A)* {
            precedence
            L + -
            L * /
        }
    )";

    parser.log = [](size_t line, size_t col, const std::string & msg) {
        MY_USER_ERROR(msg << " ( LINE: " << line << " COLUMN: " << col << ")")
    };

    bool success = parser.load_grammar(grammar);
    if (!success) MY_RUNTIME_ERROR("LANGUAGE GRAMMAR COULD NOT BE LOADED!")

    parser["OperatorExpression"] = [](const peg::SemanticValues & vs) {
        auto left = std::any_cast<std::shared_ptr<ASTTNode>>(vs[0]);
        if (vs.size() > 1)
        {
            auto op = std::any_cast<char>(vs[1]);
            auto right = std::any_cast<std::shared_ptr<ASTTNode>>(vs[2]);
            switch (op)
            {
                case '+':
                    left = std::make_unique<ASTTNode>(ASTNodeType::Addition, left, right);
                    break;
                case '-':
                    left = std::make_unique<ASTTNode>(ASTNodeType::Substraction, left, right);
                    break;
                case '*':
                    left = std::make_unique<ASTTNode>(ASTNodeType::Multiplication, left, right);
                    break;
                case '/':
                    left = std::make_unique<ASTTNode>(ASTNodeType::Division, left, right);
                    break;
                default: MY_RUNTIME_ERROR("PARSER ISSUE!")
            }
        }
        return left;
    };

    parser["Operator"] = [](const peg::SemanticValues & vs) { return vs.sv()[0]; };

    parser["Identifier"] = [](const peg::SemanticValues & vs) {
        return std::make_shared<ASTTNode>(std::string(vs.token()));
    };

    parser["Register"] = [](const peg::SemanticValues & vs) {
        if (vs.token() == "x")
            return std::make_shared<ASTTNode>(ASTNodeType::Reference_X);
        else if (vs.token() == "y")
            return std::make_shared<ASTTNode>(ASTNodeType::Reference_Y);
        else if (vs.token() == "z")
            return std::make_shared<ASTTNode>(ASTNodeType::Reference_Z);
        else if (vs.token() == "cv")
            return std::make_shared<ASTTNode>(ASTNodeType::Reference_CV);
        else if (vs.token() == "u")
            return std::make_shared<ASTTNode>(ASTNodeType::Reference_U);
        else if (vs.token() == "v")
            return std::make_shared<ASTTNode>(ASTNodeType::Reference_V);
        else if (vs.token() == "w")
            return std::make_shared<ASTTNode>(ASTNodeType::Reference_W);
        else
            MY_RUNTIME_ERROR("PARSER ISSUE!")
        // else if (vs.token() == "f0")
        //     return std::make_shared<ASTTNode>(ASTNodeType::Reference_F0);
        // else if (vs.token() == "f1")
        //     return std::make_shared<ASTTNode>(ASTNodeType::Reference_F1);
        // else if (vs.token() == "f2")
        //     return std::make_shared<ASTTNode>(ASTNodeType::Reference_F2);
        // else if (vs.token() == "f3")
        //     return std::make_shared<ASTTNode>(ASTNodeType::Reference_F3);
        // else if (vs.token() == "f4")
        //     return std::make_shared<ASTTNode>(ASTNodeType::Reference_F4);
        // else if (vs.token() == "f5")
        //     return std::make_shared<ASTTNode>(ASTNodeType::Reference_F5);
        // else if (vs.token() == "f6")
        //     return std::make_shared<ASTTNode>(ASTNodeType::Reference_F6);
        // else if (vs.token() == "f7")
        //     return std::make_shared<ASTTNode>(ASTNodeType::Reference_F7);
        // else if (vs.token() == "c0")
        //     return std::make_shared<ASTTNode>(ASTNodeType::Reference_C0);
        // else if (vs.token() == "c1")
        //     return std::make_shared<ASTTNode>(ASTNodeType::Reference_C1);
        // else if (vs.token() == "c2")
        //     return std::make_shared<ASTTNode>(ASTNodeType::Reference_C2);
        // else if (vs.token() == "c3")
        //     return std::make_shared<ASTTNode>(ASTNodeType::Reference_C3);
    };

    parser["OldRegister"] = [](const peg::SemanticValues & vs) {
        MY_USER_ERROR("Identifier '"
                      << vs.token()
                      << "' is no longer allowed, see documentation for more information.")
    };

    parser["FunctionExpression"] = [](const peg::SemanticValues & vs) {
        auto offset = static_cast<uint32_t>(granite::ASTNodeType::Operation_SQRT);

        if (vs.choice() == static_cast<uint32_t>(ASTNodeType::Operation_ARCTANGENS_2) - offset)
        {
            return std::make_shared<ASTTNode>(static_cast<ASTNodeType>(vs.choice() + offset),
                                              std::any_cast<std::shared_ptr<ASTTNode>>(vs[0]),
                                              std::any_cast<std::shared_ptr<ASTTNode>>(vs[1]));
        }
        else
        {
            return std::make_shared<ASTTNode>(static_cast<ASTNodeType>(vs.choice() + offset),
                                              std::any_cast<std::shared_ptr<ASTTNode>>(vs[0]),
                                              nullptr);
        }
    };

    parser["Constant"] = [](const peg::SemanticValues & vs) {
        return std::make_shared<ASTTNode>(vs.token_to_number<float>());
    };

    // (4) Parse
    parser.enable_packrat_parsing(); // Enable packrat parsing.

    std::shared_ptr<ASTTNode> tree;
    parser.parse(code, tree);

    return tree;
}

void granite::TrajectoryIntegrator::initializeData()
{
    MY_VLOG("INITIALIZING DATA ... ")

    if (_settings.Space == Space::Space3D)
        loadTexture3D(0, _loader.windfield());
    else
        loadTexture2D(0, _loader.windfield());

    if (_settings.AdditionalVolumeMode != granite::AdditionalVolumeMode::Off)
    {
        auto map = _loader.additionalVolumes();
        if (map.size() > MAX_ADDITIONAL_VOLUMES)
            MY_USER_WARNING("You provided more additional volumes than supported.");

        uint32_t idx = 0;
        for (auto & p : map)
        {
            auto key = p.first;
            auto field = p.second;

            if (idx == MAX_ADDITIONAL_VOLUMES) break;

            _env[key] = static_cast<granite::ASTNodeType>(
                static_cast<uint32_t>(ASTNodeType::Reference_F0) + idx);

            _keys.push_back(key);

            if (_settings.Space == Space::Space3D)
            {
                loadTexture3D(_additional_textures[0][idx].Object,      //
                              _additional_textures[0][idx].CudaArray,   //
                              _additional_textures[0][idx].Initialized, //
                              field);
            }
            else
            {
                loadTexture2D(_additional_textures[0][idx].Object,      //
                              _additional_textures[0][idx].LinearArray, //
                              _additional_textures[0][idx].Pitch,       //
                              _additional_textures[0][idx].Initialized, //
                              field, false);
            }

            idx++;
        }
        _num_additional_volumes = idx;
    }

    if (_settings.ConstantsMode != granite::ConstantsMode::Off)
    {
        auto map = _loader.constants();
        if (map.size() > MAX_CONSTANT_ADDITIONAL_COMPUTE)
            MY_USER_WARNING("You provided more constants than supported.");

        uint32_t idx = 0;
        for (auto & p : map)
        {
            auto key = p.first;
            auto field = p.second;
            pybind11::buffer_info buffer{field.request()};

            if (idx == MAX_CONSTANT_ADDITIONAL_COMPUTE) break;

            _env[key] = static_cast<granite::ASTNodeType>(
                static_cast<uint32_t>(ASTNodeType::Reference_C0) + idx);

            cudaMalloc(&_constants_device[0][idx], _set->numberTrajectories() * sizeof(float));
            MY_CUDA_ERROR_CHECK

            cudaMemcpy(_constants_device[0][idx],                  //
                       reinterpret_cast<float *>(buffer.ptr),      //
                       _set->numberTrajectories() * sizeof(float), //
                       cudaMemcpyHostToDevice);
            MY_CUDA_ERROR_CHECK

            idx++;
        }

        _num_additional_constants = idx;
    }

    if (_settings.UpLiftMode != granite::UpLiftMode::Off)
    {
        auto field = _loader.uplift();
        pybind11::buffer_info buffer{field.request()};
        if (buffer.shape[0] != _set->numberTrajectories())
            MY_USER_ERROR("Uplift dimension does not match number of particles.")

        const size_t DIM = _settings.Space == granite::Space::Space2D ? 2 : 3;

        cudaMalloc(&_uplift_device[0], _set->numberTrajectories() * sizeof(float) * DIM);
        MY_CUDA_ERROR_CHECK

        cudaMemcpy(_uplift_device[0],                                //
                   reinterpret_cast<float *>(buffer.ptr),            //
                   _set->numberTrajectories() * sizeof(float) * DIM, //
                   cudaMemcpyHostToDevice);
        MY_CUDA_ERROR_CHECK
    }

    if ((_settings.WindfieldMode == granite::WindfieldMode::Dynamic ||
         _settings.UpLiftMode == granite::UpLiftMode::Dynamic ||
         _settings.AdditionalVolumeMode == granite::AdditionalVolumeMode::Dynamic ||
         _settings.ConstantsMode == granite::ConstantsMode::Dynamic))
    {
        if (_loader.step())
            updateData(1);
        else
            MY_USER_ERROR("Cannot increment loader.");
    }

    MY_VLOG("INITIALIZING DATA ... DONE")
}

void granite::TrajectoryIntegrator::updateData(size_t index)
{
    if (_settings.WindfieldMode == granite::WindfieldMode::Dynamic)
    {
        if (_settings.Space == Space::Space3D)
            loadTexture3D(index, _loader.windfield());
        else
            loadTexture2D(index, _loader.windfield());
    }

    if (_settings.AdditionalVolumeMode == granite::AdditionalVolumeMode::Dynamic)
    {
        auto map = _loader.additionalVolumes();

        for (auto & p : map)
        {
            auto key = p.first;
            auto field = p.second;

            if (_env.find(key) == _env.end()) continue;

            uint32_t idx = static_cast<uint32_t>(_env[key]) -
                           static_cast<uint32_t>(granite::ASTNodeType::Reference_F0);

            if (_settings.Space == Space::Space3D)
            {
                loadTexture3D(_additional_textures[index][idx].Object,      //
                              _additional_textures[index][idx].CudaArray,   //
                              _additional_textures[index][idx].Initialized, //
                              field);
            }
            else
            {
                loadTexture2D(_additional_textures[index][idx].Object,      //
                              _additional_textures[index][idx].LinearArray, //
                              _additional_textures[index][idx].Pitch,       //
                              _additional_textures[index][idx].Initialized, //
                              field, false);
            }

            idx++;
        }
    }

    if (_settings.ConstantsMode == granite::ConstantsMode::Dynamic)
    {
        auto map = _loader.constants();
        if (map.size() > MAX_CONSTANT_ADDITIONAL_COMPUTE)
            MY_USER_WARNING("You provided more constants than supported.");

        for (auto & p : map)
        {
            auto key = p.first;
            auto field = p.second;
            pybind11::buffer_info buffer{field.request()};

            if (_env.find(key) == _env.end()) continue;

            uint32_t idx = static_cast<uint32_t>(_env[key]) -
                           static_cast<uint32_t>(granite::ASTNodeType::Reference_F0);

            if (!_constants_device[index][idx])
            {
                cudaMalloc(&_constants_device[index][idx],
                           _set->numberTrajectories() * sizeof(float));
                MY_CUDA_ERROR_CHECK
            }

            cudaMemcpy(_constants_device[index][idx],              //
                       reinterpret_cast<float *>(buffer.ptr),      //
                       _set->numberTrajectories() * sizeof(float), //
                       cudaMemcpyHostToDevice);
            MY_CUDA_ERROR_CHECK

            idx++;
        }
    }

    if (_settings.UpLiftMode == granite::UpLiftMode::Dynamic)
    {
        auto field = _loader.uplift();
        pybind11::buffer_info buffer{field.request()};
        if (buffer.shape[0] != _set->numberTrajectories())
            MY_USER_ERROR("Uplift dimension does not match number of particles.")

        const size_t DIM = _settings.Space == granite::Space::Space2D ? 2 : 3;

        if (!_uplift_device[index])
        {
            cudaMalloc(&_uplift_device[index], _set->numberTrajectories() * sizeof(float) * DIM);
            MY_CUDA_ERROR_CHECK
        }

        cudaMemcpy(_uplift_device[index],                            //
                   reinterpret_cast<float *>(buffer.ptr),            //
                   _set->numberTrajectories() * sizeof(float) * DIM, //
                   cudaMemcpyHostToDevice);
        MY_CUDA_ERROR_CHECK
    }
}

granite::TrajectoryIntegrator::~TrajectoryIntegrator()
{
    if (_settings.Space == Space::Space3D)
    {
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                if (_windfield_textures[i][j].Initialized)
                    cudaFreeArray(_windfield_textures[i][j].CudaArray);
            }
        }

        for (size_t j = 0; j < 3; ++j)
        {
            for (size_t i = 0; i < MAX_ADDITIONAL_VOLUMES; ++i)
            {
                if (_additional_textures[j][i].Initialized)
                {
                    cudaFreeArray(_additional_textures[j][i].CudaArray);
                }
            }
        }
    }
    else
    {
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 2; ++j)
            {
                if (_windfield_textures[i][j].Initialized)
                    cudaFree(_windfield_textures[i][j].LinearArray);
            }
        }

        for (size_t j = 0; j < 3; ++j)
        {
            for (size_t i = 0; i < MAX_ADDITIONAL_VOLUMES; ++i)
            {
                if (_additional_textures[j][i].Initialized)
                {
                    cudaFree(_additional_textures[j][i].LinearArray);
                }
            }
        }
    }

    for (size_t i = 0; i < 3; ++i)
    {
        if (_uplift_device[i]) cudaFree(_uplift_device[i]);

        for (size_t j = 0; j < MAX_CONSTANT_ADDITIONAL_COMPUTE; ++j)
        {
            if (_constants_device[i][j]) cudaFree(_constants_device[i][j]);
        }
    }

    if (_topography_texture.Initialized) cudaFree(_topography_texture.LinearArray);
    if (_particles_device.front()) cudaFree(_particles_device.front());
    if (_particles_device.back()) cudaFree(_particles_device.back());
}

void granite::TrajectoryIntegrator::initSet(granite::TrajectorySet & set)
{
    // create the internal set
    _set = std::make_unique<TrajectorySet>(set._space, set._num_trajectories);

    // and load the last points from the provided set
    if (_set->_space == Space::Space2D)
        _set->add_(set._trajectories2d[set._trajectories2d.size() - 1]);
    else
        _set->add_(set._trajectories3d[set._trajectories3d.size() - 1]);

    if (_settings.Space != _set->_space)
        MY_USER_ERROR("Dimensions of IntegratorSettings and provided TrajectorySet do not match.");

    if (_set->_space == Space::Space2D)
        _start_particles_2d = _set->_trajectories2d[_set->_trajectories2d.size() - 1];
    else
        _start_particles_3d = _set->_trajectories3d[_set->_trajectories3d.size() - 1];
}

void granite::TrajectoryIntegrator::loadTexture3D(cudaTextureObject_t & tex_obj, //
                                                  cudaArray_t & array,           //
                                                  bool & initialized,            //
                                                  const pybind11::array_t<float> & data)
{
    MY_VLOG("INITIALIZING 3D TEXTURE ... ")

    pybind11::buffer_info buffer = data.request();

    if (buffer.shape.size() != 3)
        MY_USER_ERROR("Invalid dataset was provided. Wrong amout of data (3D).");

    if (!_dimensions_loaded)
    {
        _grid.Z = static_cast<size_t>(buffer.shape[0]);
        _grid.Y = static_cast<size_t>(buffer.shape[1]);
        _grid.X = static_cast<size_t>(buffer.shape[2]);
        _dimensions_loaded = true;
    }

    if (static_cast<size_t>(buffer.shape[0]) != _grid.Z)
        MY_USER_ERROR("Invalid dataset was provided. Shape does not match in z dimension (3D).");
    if (static_cast<size_t>(buffer.shape[1]) != _grid.Y)
        MY_USER_ERROR("Invalid dataset was provided. Shape does not match in y dimension (3D).");
    if (static_cast<size_t>(buffer.shape[2]) != _grid.X)
        MY_USER_ERROR("Invalid dataset was provided. Shape does not match in x dimension (3D).");

    float * cbuffer = reinterpret_cast<float *>(buffer.ptr);

    const cudaExtent extent = make_cudaExtent(_grid.X, _grid.Y, _grid.Z);
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
    MY_CUDA_ERROR_CHECK

    if (!initialized)
    {
        cudaMalloc3DArray(&array, &channel_desc, extent);
        MY_CUDA_ERROR_CHECK
    }

    cudaMemcpy3DParms copy_params{0};
    copy_params.srcPtr = make_cudaPitchedPtr(reinterpret_cast<void *>(cbuffer), //
                                             extent.width * sizeof(float),      //
                                             extent.width,                      //
                                             extent.height);
    copy_params.dstArray = array;
    copy_params.extent = extent;
    copy_params.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copy_params);
    MY_CUDA_ERROR_CHECK

    if (!initialized)
    {
        cudaResourceDesc tex_res;
        memset(&tex_res, 0, sizeof(cudaResourceDesc));
        tex_res.resType = cudaResourceTypeArray;
        tex_res.res.array.array = array;

        cudaTextureDesc tex_descr;
        memset(&tex_descr, 0, sizeof(cudaTextureDesc));
        tex_descr.normalizedCoords = false;
        tex_descr.filterMode = cudaFilterModeLinear;
        for (size_t i = 0; i < 3; ++i) tex_descr.addressMode[i] = cudaAddressModeBorder;
        tex_descr.readMode = cudaReadModeElementType;
        cudaCreateTextureObject(&tex_obj, &tex_res, &tex_descr, NULL);
        MY_CUDA_ERROR_CHECK

        initialized = true;
    }

    MY_VLOG("INITIALIZING 3D TEXTURE ... DONE");
}

void granite::TrajectoryIntegrator::loadTexture2D(cudaTextureObject_t & tex_obj,         //
                                                  float *& device_buffer,                //
                                                  size_t & pitch,                        //
                                                  bool & initialized,                    //
                                                  const pybind11::array_t<float> & data, //
                                                  bool load_dimension)
{
    MY_VLOG("INITIALIZING 2D TEXTURE ...");

    pybind11::buffer_info buffer = data.request();

    if (load_dimension)
    {
        if (buffer.shape.size() != 2)
            MY_USER_ERROR("Invalid dataset was provided. Wrong amout of data (2D).");

        if (!_dimensions_loaded)
        {
            _grid.Y = static_cast<size_t>(buffer.shape[0]);
            _grid.X = static_cast<size_t>(buffer.shape[1]);
            _dimensions_loaded = true;
        }

        if (static_cast<size_t>(buffer.shape[0]) != _grid.Y)
            MY_USER_ERROR(
                "Invalid dataset was provided. Shape does not match in y dimension (2D).");
        if (static_cast<size_t>(buffer.shape[1]) != _grid.X)
            MY_USER_ERROR(
                "Invalid dataset was provided. Shape does not match in x dimension (2D).");
    }

    float * cbuffer = reinterpret_cast<float *>(buffer.ptr);

    if (!initialized)
    {
        cudaMallocPitch(&device_buffer, &pitch, sizeof(float) * _grid.X, _grid.Y);
        MY_CUDA_ERROR_CHECK
    }

    cudaMemcpy2D(device_buffer, pitch, cbuffer, sizeof(float) * _grid.X, sizeof(float) * _grid.X,
                 _grid.Y, cudaMemcpyHostToDevice);
    MY_CUDA_ERROR_CHECK

    if (!initialized)
    {
        cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType = cudaResourceTypePitch2D;
        res_desc.res.pitch2D.devPtr = device_buffer;
        res_desc.res.pitch2D.pitchInBytes = pitch;
        res_desc.res.pitch2D.width = _grid.X;
        res_desc.res.pitch2D.height = _grid.Y;
        res_desc.res.pitch2D.desc = cudaCreateChannelDesc<float>();

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = false;
        texDesc.filterMode = cudaFilterModeLinear;

        cudaCreateTextureObject(&tex_obj, &res_desc, &texDesc, NULL);
        MY_CUDA_ERROR_CHECK

        initialized = true;
    }

    MY_VLOG("INITIALIZING 2D TEXTURE ... DONE");
}

void granite::TrajectoryIntegrator::loadTexture3D(
    size_t texture_index, const std::vector<pybind11::array_t<float>> & data)
{
    for (size_t dim = 0; dim < 3; ++dim)
    {
        loadTexture3D(_windfield_textures[texture_index][dim].Object,      //
                      _windfield_textures[texture_index][dim].CudaArray,   //
                      _windfield_textures[texture_index][dim].Initialized, //
                      data[dim]);
    }
}

void granite::TrajectoryIntegrator::loadTexture2D(
    size_t texture_index, const std::vector<pybind11::array_t<float>> & data)
{
    for (size_t dim = 0; dim < 2; ++dim)
    {
        loadTexture2D(_windfield_textures[texture_index][dim].Object,      //
                      _windfield_textures[texture_index][dim].LinearArray, //
                      _windfield_textures[texture_index][dim].Pitch,       //
                      _windfield_textures[texture_index][dim].Initialized, //
                      data[dim],                                           //
                      true);
    }
}

void granite::TrajectoryIntegrator::copyParticlePositions(const size_t NUM_POINTS,
                                                          my::math::Vec3 * device_points)
{
    // gather new trajectory points
    std::vector<my::math::Vec3> step_data(NUM_POINTS);
    cudaMemcpy(step_data.data(), device_points, NUM_POINTS * sizeof(my::math::Vec3),
               cudaMemcpyDeviceToHost);
    MY_CUDA_ERROR_CHECK

    _set->_trajectories3d.push_back(step_data);
}

void granite::TrajectoryIntegrator::copyParticlePositions(const size_t NUM_POINTS,
                                                          my::math::Vec2 * device_points)
{
    // gather new trajectory points
    std::vector<my::math::Vec2> step_data(NUM_POINTS);
    cudaMemcpy(step_data.data(), device_points, NUM_POINTS * sizeof(my::math::Vec2),
               cudaMemcpyDeviceToHost);
    MY_CUDA_ERROR_CHECK

    _set->_trajectories2d.push_back(step_data);
}

void granite::TrajectoryIntegrator::initializeTopography()
{
    MY_VLOG("INITIALIZING TOPOGRAPHY ...");

    if (_settings.Topography.size() > 0)
    {
        const pybind11::array_t<float> & data{_settings.Topography};
        pybind11::buffer_info buffer = data.request();

        if (buffer.shape.size() != 2)
            MY_USER_ERROR("Invalid topography was provided. Wrong dimension of data (2D).");

        size_t x = static_cast<size_t>(buffer.shape[1]);
        size_t y = static_cast<size_t>(buffer.shape[0]);

        if (x != _grid.X || y != _grid.Y)
            MY_USER_ERROR("Topography dimensions does not match to provided data!");

        loadTexture2D(_topography_texture.Object,      //
                      _topography_texture.LinearArray, //
                      _topography_texture.Pitch,       //
                      _topography_texture.Initialized, //
                      data,                            //
                      false // No dimension check, as its done above already
        );
    }

    MY_VLOG("INITIALIZING TOPOGRAPHY ... DONE");
}

void granite::TrajectoryIntegrator::verifySettings()
{
    if (_settings.Space == Space::Space3D) // 3D specific verifications
    {
        if (_settings.BorderMode != BorderMode::Block)
            MY_USER_WARNING("BorderMode is not set to block, but running simulations in 3D space!")

        if (_settings.SphereRadius > 0)
            MY_USER_WARNING("SphereRadius is not <= 0.f, but running simulations in 3D space!")
    }
    else // 2D
    {
        if (_settings.CurvatureMode == CurvatureMode::IndividualAndTotalCurvature ||
            _settings.CurvatureMode == CurvatureMode::TotalCurvature)
        {
            MY_USER_ERROR("Invalid curvature mode for 2D space!")
        }
    }

    MY_VLOG("SETTINGS VERIFICATION ... DONE");
}

std::unique_ptr<granite::TrajectorySet> granite::TrajectoryIntegrator::compute()
{
    using namespace my::math;

    _simulation_counter = 0;
    return _settings.Space == Space::Space3D ? std::move(compute3D()) : std::move(compute2D());
}

float granite::interpret(const std::shared_ptr<granite::ASTTNode> & node, const float * ref)
{
    if (node == nullptr) return 0.f;

    switch (node->Type)
    {
        case ASTNodeType::Reference_X: return ref[0];
        case ASTNodeType::Reference_Y: return ref[1];
        case ASTNodeType::Reference_Z: return ref[2];
        case ASTNodeType::Reference_CV: return ref[3];
        case ASTNodeType::Reference_F0: return ref[4];
        case ASTNodeType::Reference_F1: return ref[5];
        case ASTNodeType::Reference_F2: return ref[6];
        case ASTNodeType::Reference_F3: return ref[7];
        case ASTNodeType::Reference_F4: return ref[8];
        case ASTNodeType::Reference_F5: return ref[9];
        case ASTNodeType::Reference_F6: return ref[10];
        case ASTNodeType::Reference_F7: return ref[11];
        case ASTNodeType::Reference_C0: return ref[12];
        case ASTNodeType::Reference_C1: return ref[13];
        case ASTNodeType::Reference_C2: return ref[14];
        case ASTNodeType::Reference_C3: return ref[15];
        case ASTNodeType::Reference_U: return ref[16];
        case ASTNodeType::Reference_V: return ref[17];
        case ASTNodeType::Reference_W: return ref[18];
        case ASTNodeType::Constant: return node->Value;
        case ASTNodeType::Addition: return interpret(node->Left, ref) + interpret(node->Right, ref);
        case ASTNodeType::Multiplication:
            return interpret(node->Left, ref) * interpret(node->Right, ref);
        case ASTNodeType::Substraction:
            return interpret(node->Left, ref) - interpret(node->Right, ref);
        case ASTNodeType::Division: { // todo
            return interpret(node->Left, ref) / interpret(node->Right, ref);
        }
        case ASTNodeType::Operation_SQRT: return sqrtf(interpret(node->Left, ref));
        case ASTNodeType::Operation_EXP: return expf(interpret(node->Left, ref));
        case ASTNodeType::Operation_LN: return logf(interpret(node->Left, ref));
        case ASTNodeType::Operation_SINE: return sinf(interpret(node->Left, ref));
        case ASTNodeType::Operation_COSINE: return cosf(interpret(node->Left, ref));
        case ASTNodeType::Operation_TANGENS: return tanf(interpret(node->Left, ref));
        case ASTNodeType::Operation_ARCSINE: return asinf(interpret(node->Left, ref));
        case ASTNodeType::Operation_ARCCOSINE: return acosf(interpret(node->Left, ref));
        case ASTNodeType::Operation_ARCTANGENS: return atanf(interpret(node->Left, ref));
        case ASTNodeType::Operation_ARCTANGENS_2:
            return atan2f(interpret(node->Left, ref), interpret(node->Right, ref));
        default: return 0.f;
    }
}

std::shared_ptr<granite::ASTTNode>
granite::replace_environment(std::shared_ptr<granite::ASTTNode> node,
                             const std::unordered_map<std::string, ASTNodeType> & env)
{
    if (node->Type == ASTNodeType::Identifier)
    {
        if (auto type = env.find(node->Id); type != env.end())
        {
            node->Type = std::get<1>(*type);
            node->Value = 0.f;
            node->Left = nullptr;
            node->Right = nullptr;
        }
        else
        {
            MY_USER_ERROR("Invalid identifier '" << node->Id << "'.");
        }
    }

    if (node->Left) node->Left = replace_environment(node->Left, env);

    if (node->Right) node->Right = replace_environment(node->Right, env);

    return node;
}

std::vector<granite::ASTANode> granite::flatten(const std::shared_ptr<granite::ASTTNode> & node)
{
    if (node == nullptr) return std::vector<ASTANode>();

    auto left = flatten(node->Left);
    auto right = flatten(node->Right);

    int offset = 1;
    for (size_t i = 0; i < left.size(); ++i)
    {
        if (left[i].Left != -1) left[i].Left += offset;
        if (left[i].Right != -1) left[i].Right += offset;
    }

    offset += static_cast<int>(left.size());
    for (size_t i = 0; i < right.size(); ++i)
    {
        if (right[i].Left != -1) right[i].Left += offset;
        if (right[i].Right != -1) right[i].Right += offset;
    }

    std::vector<ASTANode> result{
        {node->Type, node->Value, node->Left ? 1 : -1, node->Right ? offset : -1}};

    result.insert(result.end(), left.begin(), left.end());
    result.insert(result.end(), right.begin(), right.end());

    return result;
}

std::ostream & granite::operator<<(std::ostream & os,
                                   const std::shared_ptr<granite::ASTTNode> & node)
{
    if (node == nullptr) return os << "<NULL NODE>";

    switch (node->Type)
    {
        case ASTNodeType::Reference_X: os << "x"; break;
        case ASTNodeType::Reference_Y: os << "y"; break;
        case ASTNodeType::Reference_Z: os << "z"; break;
        case ASTNodeType::Reference_CV: os << "cv"; break;
        case ASTNodeType::Reference_F1: os << "f1"; break;
        case ASTNodeType::Reference_F2: os << "f2"; break;
        case ASTNodeType::Reference_F3: os << "f3"; break;
        case ASTNodeType::Reference_F4: os << "f4"; break;
        case ASTNodeType::Reference_F5: os << "f5"; break;
        case ASTNodeType::Reference_F6: os << "f6"; break;
        case ASTNodeType::Reference_F7: os << "f7"; break;
        case ASTNodeType::Reference_C0: os << "c0"; break;
        case ASTNodeType::Reference_C1: os << "c1"; break;
        case ASTNodeType::Reference_C2: os << "c2"; break;
        case ASTNodeType::Reference_C3: os << "c3"; break;
        case ASTNodeType::Reference_U: os << "u"; break;
        case ASTNodeType::Reference_V: os << "v"; break;
        case ASTNodeType::Reference_W: os << "w"; break;
        case ASTNodeType::Identifier: os << node->Id; break;
        case ASTNodeType::Constant: os << node->Value; break;
        case ASTNodeType::Addition: os << "(" << node->Left << "+" << node->Right << ")"; break;
        case ASTNodeType::Multiplication:
            os << "(" << node->Left << "*" << node->Right << ")";
            break;
        case ASTNodeType::Substraction: os << "(" << node->Left << "-" << node->Right << ")"; break;
        case ASTNodeType::Division: os << "(" << node->Left << "/" << node->Right << ")"; break;
        case ASTNodeType::Operation_SQRT: os << "sqrt(" << node->Left << ")"; break;
        case ASTNodeType::Operation_EXP: os << "exp(" << node->Left << ")"; break;
        case ASTNodeType::Operation_LN: os << "ln(" << node->Left << ")"; break;
        case ASTNodeType::Operation_SINE: os << "sin(" << node->Left << ")"; break;
        case ASTNodeType::Operation_COSINE: os << "cos(" << node->Left << ")"; break;
        case ASTNodeType::Operation_TANGENS: os << "cos(" << node->Left << ")"; break;
        case ASTNodeType::Operation_ARCSINE: os << "asin(" << node->Left << ")"; break;
        case ASTNodeType::Operation_ARCCOSINE: os << "acos(" << node->Left << ")"; break;
        case ASTNodeType::Operation_ARCTANGENS: os << "atan(" << node->Left << ")"; break;
        case ASTNodeType::Operation_ARCTANGENS_2:
            os << "atan2(" << node->Left << ", " << node->Right << ")";
            break;
        default: os << "<INVALID NODE>"; break;
    }

    return os;
}

void granite::_print(const ASTANode * nodes, const size_t num_nodes, const size_t i)
{
    if (i >= num_nodes)
    {
        printf("<INVALID INDEX>");
        return; // todo
    }

    auto node{nodes[i]};
    switch (node.Type)
    {
        case ASTNodeType::Reference_X: printf("x"); break;
        case ASTNodeType::Reference_Y: printf("y"); break;
        case ASTNodeType::Reference_Z: printf("z"); break;
        case ASTNodeType::Reference_CV: printf("cv"); break;
        case ASTNodeType::Reference_F0: printf("f0"); break;
        case ASTNodeType::Reference_F1: printf("f1"); break;
        case ASTNodeType::Reference_F2: printf("f2"); break;
        case ASTNodeType::Reference_F3: printf("f3"); break;
        case ASTNodeType::Reference_F4: printf("f4"); break;
        case ASTNodeType::Reference_F5: printf("f5"); break;
        case ASTNodeType::Reference_F6: printf("f6"); break;
        case ASTNodeType::Reference_F7: printf("f7"); break;
        case ASTNodeType::Reference_C0: printf("c0"); break;
        case ASTNodeType::Reference_C1: printf("c1"); break;
        case ASTNodeType::Reference_C2: printf("c2"); break;
        case ASTNodeType::Reference_C3: printf("c3"); break;
        case ASTNodeType::Reference_U: printf("u"); break;
        case ASTNodeType::Reference_V: printf("v"); break;
        case ASTNodeType::Reference_W: printf("w"); break;
        case ASTNodeType::Identifier: MY_RUNTIME_ERROR("Found identifier in flattened ast.");
        case ASTNodeType::Constant: printf("%f", node.Value); break;
        case ASTNodeType::Addition: {
            printf("(");
            _print(nodes, num_nodes, node.Left);
            printf("+");
            _print(nodes, num_nodes, node.Right);
            printf(")");
        }
        break;
        case ASTNodeType::Multiplication: {
            printf("(");
            _print(nodes, num_nodes, node.Left);
            printf("*");
            _print(nodes, num_nodes, node.Right);
            printf(")");
        }
        break;
        case ASTNodeType::Substraction: {
            printf("(");
            _print(nodes, num_nodes, node.Left);
            printf("-");
            _print(nodes, num_nodes, node.Right);
            printf(")");
        }
        break;
        case ASTNodeType::Division: {
            printf("(");
            _print(nodes, num_nodes, node.Left);
            printf("/");
            _print(nodes, num_nodes, node.Right);
            printf(")");
        }
        break;
        case ASTNodeType::Operation_SQRT: {
            printf("sqrt(");
            _print(nodes, num_nodes, node.Left);
            printf(")");
        }
        break;
        case ASTNodeType::Operation_EXP: {
            printf("exp(");
            _print(nodes, num_nodes, node.Left);
            printf(")");
        }
        break;
        case ASTNodeType::Operation_LN: {
            printf("ln(");
            _print(nodes, num_nodes, node.Left);
            printf(")");
        }
        break;
        case ASTNodeType::Operation_SINE: {
            printf("sin(");
            _print(nodes, num_nodes, node.Left);
            printf(")");
        }
        break;
        case ASTNodeType::Operation_COSINE: {
            printf("cos(");
            _print(nodes, num_nodes, node.Left);
            printf(")");
        }
        break;
        case ASTNodeType::Operation_TANGENS: {
            printf("tan(");
            _print(nodes, num_nodes, node.Left);
            printf(")");
        }
        break;
        case ASTNodeType::Operation_ARCSINE: {
            printf("asin(");
            _print(nodes, num_nodes, node.Left);
            printf(")");
        }
        break;
        case ASTNodeType::Operation_ARCCOSINE: {
            printf("acos(");
            _print(nodes, num_nodes, node.Left);
            printf(")");
        }
        break;
        case ASTNodeType::Operation_ARCTANGENS: {
            printf("atan(");
            _print(nodes, num_nodes, node.Left);
            printf(")");
        }
        break;
        case ASTNodeType::Operation_ARCTANGENS_2: {
            printf("isqrt(");
            _print(nodes, num_nodes, node.Left);
            printf(", ");
            _print(nodes, num_nodes, node.Right);
            printf(")");
        }
        break;
        default: printf("<INVALID NODE>");
    }
}

void granite::print(const ASTANode * nodes, const size_t num_nodes)
{
    _print(nodes, num_nodes, 0);
    printf("\n");
}

bool granite::contains(const std::shared_ptr<ASTTNode> & node, ASTNodeType t)
{
    if (node->Type == t) return true;
    bool result{false};
    if (node->Left) result |= contains(node->Left, t);
    if (node->Right) result |= contains(node->Right, t);
    return result;
}