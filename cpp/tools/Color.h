#ifndef FMFUSOIN_COLOR_H
#define FMFUSOIN_COLOR_H


namespace fmfusion
{
    // Generate 40 colors as a color bar
    std::array<Eigen::Vector3d, 40> InstanceColorBar40 =
    {
        Eigen::Vector3d(0.0, 0.0, 1.0), 
        Eigen::Vector3d(1.0, 1.0, 0.0), 
        Eigen::Vector3d(1.0, 0.0, 1.0), 
        Eigen::Vector3d(0.0, 1.0, 1.0),
        Eigen::Vector3d(0.5, 0.0, 0.0), 
        Eigen::Vector3d(0.0, 0.5, 0.0), 
        Eigen::Vector3d(0.0, 0.0, 0.5), 
        Eigen::Vector3d(0.5, 0.5, 0.0), 
        Eigen::Vector3d(0.5, 0.0, 0.5), 
        Eigen::Vector3d(0.0, 0.5, 0.5),
        Eigen::Vector3d(0.75, 0.0, 0.0), 
        Eigen::Vector3d(0.0, 0.75, 0.0), 
        Eigen::Vector3d(0.0, 0.0, 0.75), 
        Eigen::Vector3d(0.75, 0.75, 0.0), 
        Eigen::Vector3d(0.75, 0.0, 0.75), 
        Eigen::Vector3d(0.0, 0.75, 0.75),
        Eigen::Vector3d(0.25, 0.0, 0.0), 
        Eigen::Vector3d(0.0, 0.25, 0.0), 
        Eigen::Vector3d(0.0, 0.0, 0.25), 
        Eigen::Vector3d(0.25, 0.25, 0.0), 
        Eigen::Vector3d(0.25, 0.0, 0),
        Eigen::Vector3d(0.0, 0.25, 0.25),
        Eigen::Vector3d(0.375, 0.0, 0.0),
        Eigen::Vector3d(0.0, 0.375, 0.0),
        Eigen::Vector3d(0.0, 0.0, 0.375),
        Eigen::Vector3d(0.375, 0.375, 0.0),
        Eigen::Vector3d(0.375, 0.0, 0.375),
        Eigen::Vector3d(0.0, 0.375, 0.375),
        Eigen::Vector3d(0.125, 0.0, 0.0),
        Eigen::Vector3d(0.0, 0.125, 0.0),
        Eigen::Vector3d(0.0, 0.0, 0.125),
        Eigen::Vector3d(0.125, 0.125, 0.0),
        Eigen::Vector3d(0.125, 0.0, 0.125),
        Eigen::Vector3d(0.0, 0.125, 0.125),
        Eigen::Vector3d(0.625, 0.0, 0.0),
        Eigen::Vector3d(0.0, 0.625, 0.0),
        Eigen::Vector3d(0.0, 0.0, 0.625),
        Eigen::Vector3d(0.625, 0.625, 0.0),
        Eigen::Vector3d(0.625, 0.0, 0.625),
        Eigen::Vector3d(0.0, 0.625, 0.625)
    };


    std::array<Eigen::Vector3d,20> InstanceColorBar20 = 
    {
        Eigen::Vector3d(0.65098039, 0.80784314, 0.89019608),
        Eigen::Vector3d(0.12156863, 0.47058824, 0.70588235),
        Eigen::Vector3d(0.69803922, 0.8745098 , 0.54117647),
        Eigen::Vector3d(0.2       , 0.62745098, 0.17254902),
        Eigen::Vector3d(0.98431373, 0.60392157, 0.6       ),
        Eigen::Vector3d(0.89019608, 0.10196078, 0.10980392),
        Eigen::Vector3d(0.99215686, 0.74901961, 0.43529412),
        Eigen::Vector3d(1.        , 0.49803922, 0.        ),
        Eigen::Vector3d(0.79215686, 0.69803922, 0.83921569),
        Eigen::Vector3d(0.41568627, 0.23921569, 0.60392157),
        Eigen::Vector3d(1.        , 1.        , 0.6       ),
        Eigen::Vector3d(0.69411765, 0.34901961, 0.15686275),
        Eigen::Vector3d(0.97647059, 0.45098039, 0.02352941),
        Eigen::Vector3d(0.63529412, 0.07843137, 0.18431373),
        Eigen::Vector3d(0.6       , 0.9       , 0.6       ),
        Eigen::Vector3d(0.4       , 0.7       , 0.4       ),
        Eigen::Vector3d(0.6       , 0.6       , 0.6       ),
        Eigen::Vector3d(0.4       , 0.4       , 0.7       ),
        Eigen::Vector3d(0.6       , 0.6       , 0.6       ),
        Eigen::Vector3d(0.4       , 0.4       , 0.4       )
    };

}

#endif

