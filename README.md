# Raymarching 学习笔记

原文 https://github.com/electricsquare/raymarching-workshop

原作者例程列表：
- Part 1a: https://www.shadertoy.com/view/XltBzj
- Part 1b: https://www.shadertoy.com/view/4tdBzj
- Part 1c: https://www.shadertoy.com/view/4t3fzn
- Part 2:  https://www.shadertoy.com/view/Xl3fzn
- Part 3a: https://www.shadertoy.com/view/Xtcfzn
- Part 3b: https://www.shadertoy.com/view/XlcfRM
- Part 4:  https://www.shadertoy.com/view/MtdBzs

## Overview
传统光栅化方法的GPU将三角形网格作为输入，将它们光栅化为像素，然后对它们进行“着色”以计算它们对图像的贡献。这种方式目前应用最广。
另一种方法是通过每个像素投射一条射线，与场景中的表面相交，然后迭代计算光线反射，最终得出这个像素的颜色。

本文将要介绍一种通过**距离场**进行射线投影的技术——RayMarching。距离场是一个函数，这个函数能**返回一个给定的点与场景中最近的表面的距离**。这个距离定义了每个点周围空的球体的半径。同时定义在物体内部和外部的距离场（内负外正）就是有符号的距离场（Signed distance fields——SDFs）。ShaderToy大量的Demo都是基于此项技术实现。

### What's possible with raymarching
1. 轻松实现完全动态的表面拓扑结构、形状变形、高质量软阴影和环境遮蔽。这些效果用三角形网格是很难实现或者开销巨大的。
2. 通过使用SDF，场景的几何体完全以参数化的方式表示。这使得通过简单地改变对场景映射函数的输入来实现形状的动画化变得非常简单。
3. 与传统的栅格化方法相比，raymarching 使其他图形效果变得更加简单。例如，次表面散射，只需要向表面发送一些额外的光线，就可以看到它有多厚。环境遮蔽、抗锯齿和景深是另外三种技术，它们只需要额外的几条线，但却能大大改善图像质量。

### Raymarching distance fields
我们将沿着每条射线行进，寻找与场景中的表面的交点。One way to do this 是从射线的原点（摄像机）开始，**沿着射线均匀地走一个又一个step**，在每个点评估距离场。当与场景的距离小于一个阈值时，我们就知道我们已经碰到了一个表面，终止射线的行进并对该像素进行着色。

更有效的方法是**使用 SDF 返回的距离来决定下一步的大小**。SDF 返回的距离可以被看作是输入点周围空球体的半径。因此，沿着射线按这个量步进是安全的，因为我们知道我们不会通过任何表面。

在下面的光线行进的二维表示法中，每个圆圈的中心都是对场景进行采样的地方。然后射线沿着这个距离行进（延伸到圆的半径），然后再重新采样。

如你所见，对SDF进行采样并不能给你提供射线的准确交点，而是你在**不通过表面的情况下可以走的最小距离**。一旦这个距离低于设定的阈值，射线的行进就终止了，像素可以根据与之相交的表面的属性进行着色。
![](assets/Raymarching%20学习笔记/image-20221018102215763.png)
https://www.shadertoy.com/view/lslXD8

### Comparison to ray tracing
有人可能会问，为什么我们不直接使用分析数学来计算与场景的交集，使用光线追踪的技术。（光线追踪是离线渲染的典型工作方式——场景中的所有三角形都被索引到空间数据结构中，比如BVH或kD树，使得射线和三角形高效求交。）

Raymarching的优势在于：
- 实现射线投射例程是非常简单的
- 我们避免了实现射线——三角形交点和BVH数据结构的所有复杂性
- 我们不需要编写明确的场景表示——三角形网格、纹理坐标、颜色等。
- 我们可以从距离场的一系列有用的特性中获益，其中一些在上面提到过。


## Let's begin!

### 2D SDF demo
仓库原作者们提供了一个简单的框架来定义和可视化2D SDF： https://www.shadertoy.com/view/Wsf3Rj

这个项目在定义距离场之前，结果是纯白色的。我们现在的目标是设计一个SDF，给出所需的场景形状（白色轮廓）。在代码中，这个距离是由 `sdf()` 函数计算的（返回0.所以显示纯白色），它被赋予了一个2D空间的位置作为输入。你在这里学到的概念将直接推广到三维空间，并允许你为三维场景建模。
![](assets/Raymarching%20学习笔记/image-20221018110453502.png)
从简单的开始——尝试只使用输入源 `vec2 p` 的x或者y分量
```C
float sdf(vec2 p)
{
    return p.y;
}
```
现在的输出结果是：
![](assets/Raymarching%20学习笔记/image-20221018111138555.png)

我们的框架用绿色表示外面，红色表示内面，白色表示面本身。内/外区域的阴影表示距离等值线——固定距离的线（原文and the shading in the inside/outside regions illustrates distance iso-lines - lines at fixed distances.）。在二维中，这个SDF模拟了二维中 `y=0` 的一条水平线。在三维中，这将代表什么样的几何基元？

你还可以尝试使用距离因子来编写SDF，比如 `return length(p);` 通过从距离中减去所需的半径来给这个点一些面积：`return length(p) - 0.25;`。也可以在取其大小之前修改输入点：`length(p - vec2(0.0, 0.2)) - 0.25;`。
![](assets/Raymarching%20学习笔记/image-20221018112310120.png)

现在我们用数学建模造了一个圆。这种场景表示法与其他 "显式 "场景表示法（如三角形网格）进行对比，我们仅用一行代码就能创建一个球体，并且代码能直接映射到球体数学定义之一——"与中心点等距的所有点的集合"。

对于其他类型的基元，SDF函数也同样优雅。iq做了一个带图片的参考：[http://iquilezles.org/www/articles/distfunctions/distfunctions.htm](http://iquilezles.org/www/articles/distfunctions/distfunctions.htm)

当你理解了到一个基元的距离是如何工作的——把它放在一个box里并为它定义一个函数，你就不需要每次都记住和写出代码。有一个已经为圆定义的函数 `sdCircle()`，你可以在着色器中找到。添加任何你想要的基元。

### Combining shapes

现在我们知道了如何创建单个基元，那么我们如何将它们组合起来以定义一个具有多个形状的场景？

One way to do this is the 'union' operator
union 被定义为两个距离的最小值。直觉上，SDF给出的是到最近的表面的距离，如果场景中有多个物体表面，你需要取到最近的物体的距离，如代码所示。
```C
float sdf(vec2 p)
{
    float d = 1000.0;
    
    d = min(d, sdCircle(p, vec2(-0.1, 0.4), 0.15));
    d = min(d, sdCircle(p, vec2( 0.5, 0.1), 0.35));
    
    return d;
}
```

通过这种方式我们可以组合许多形状，一旦理解了这点，就应该使用 `opU()` 函数，他代表了 union操作。
我们还可以使用一个花哨的 soft min 函数获得平滑的混合——尝试使用提供的 `opBlend()`。
```C
float sdf(vec2 p)
{    
    float d1 = sdCircle(p, vec2(-0.1, 0.4), 0.15);
    float d2 = sdCircle(p, vec2( 0.5, 0.1), 0.35);
    
    return opBlend(d1,d2);
}
```

![](assets/Raymarching%20学习笔记/image-20221018134044926.png)

还有许多其他有趣的技术可以应用，感兴趣的读者可以参考这个关于用SDF构建场景的扩展介绍： [https://www.youtube.com/watch?v=s8nFqwOho-s](https://www.youtube.com/watch?v=s8nFqwOho-s)

## Transition to 3D
现在我们要开始在三维空间中进行尝试了。
本人实现的完整的shadertoy源码：[https://www.shadertoy.com/view/DslGDH](https://www.shadertoy.com/view/DslGDH)
![](assets/Raymarching%20学习笔记/image-20221018172200193.png)

### Raymarching loop
与其像我们在2D中那样将SDF可视化，不如直接进行场景的渲染。下面是我们如何实现射线行进的基本思路（伪代码）。

```
Main function
    Evaluate camera
    Call RenderRay

RenderRay function
    Raymarch to find intersection of ray with scene
    Shade
```

现在将对这些步骤分别进行更详细的描述。

### Camera
```c++
vec3 getCameraRayDir(vec2 uv, vec3 camPos, vec3 camTarget)
{
    // Calculate camera's "orthonormal basis", i.e. its transform matrix components
    vec3 camForward = normalize(camTarget - camPos);
    vec3 camRight = normalize(cross(vec3(0.0, 1.0, 0.0), camForward));
    vec3 camUp = normalize(cross(camForward, camRight));
     
    float fPersp = 3.0;
    vec3 vDir = normalize(uv.x * camRight + uv.y * camUp + camForward * fPersp);
 
    return vDir;
}
```

这个函数首先计算摄像机的 View 矩阵的三个轴：前、右和上的矢量。前进向量是指从摄像机位置到观察目标位置的归一化向量。右向量是通过将前向量与世界上的上轴交叉而找到的。向上矢量通过向前和向右的矢量叉乘获得。

最后，通过在摄像机前面取一个点，用像素坐标 `uv` 在摄像机的右边和上面方向进行偏移，来计算摄像机的光线。 `fPersp` 允许我们间接地控制摄像机的视野。你可以把这个乘法看成是把近平面移到离摄像机更近和更远的地方。

### Scene definition
```C++
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}
 
float sdf(vec3 pos)
{
    float t = sdSphere(pos-vec3(0.0, 0.0, 10.0), 3.0);
     
    return t;
}
```

如你所见，我们添加了一个 `sdSphere()` ，与 `sdCircle` 相同，只是在输入点上有一些他的组件

### Raymarching
```C++
float castRay(vec3 rayOrigin, vec3 rayDir)
{
    float t = 0.0; // Stores current distance along ray
     
    for (int i = 0; i < 64; i++)
    {
        float res = SDF(rayOrigin + rayDir * t);
        if (res < (0.0001*t))
        {
            return t;
        }
        t += res;
    }
     
    return -1.0;
}
```

我们现在将添加一个 `render` 函数，它负责为求出的交点进行着色。现在先让它显示与场景的距离，以检查我们是否步入正轨。我们将对其进行缩放和反转，以更好地看到其中的差异。
```c++
vec3 render(vec3 rayOrigin, vec3 rayDir)
{
    float t = castRay(rayOrigin, rayDir);
    
    // Visualize depth
    vec3 col = vec3(1.0-t*0.075);
    
    return col;
}
```

为了计算每条光线的方向，我们要把输入的像素坐标 `fragCoord` 从范围 `[0, w), [0, h)` 转换成 `[-a, a], [-1, 1]` ，其中 `w` 和 `h` 是屏幕的宽度和高度（像素），`a`是屏幕的纵横比。然后我们可以把返回值传给我们上面定义的 `getCameraRayDir` 函数，以获得光线方向。
```c++
vec2 normalizeScreenCoords(vec2 screenCoord)
{
    vec2 result = 2.0 * (screenCoord/iResolution.xy - 0.5);
    result.x *= iResolution.x/iResolution.y; // Correct for aspect ratio
    return result;
}
```

我们的主要图像功能看起来如下：
```C++
void mainImage(out vec4 fragColor, vec2 fragCoord)
{
    vec3 camPos = vec3(0, 0, -1);
    vec3 camTarget = vec3(0, 0, 0);
    
    vec2 uv = normalizeScreenCoords(fragCoord);
    vec3 rayDir = getCameraRayDir(uv, camPos, camTarget);   
    
    vec3 col = render(camPos, rayDir);
    
    fragColor = vec4(col, 1); // Output to screen
}
```

原作者的示例：[Part 1a](https://www.shadertoy.com/view/XltBzj)

### Ambient term
为了在场景中获得一些颜色，首先要区分物体和背景。我们可以在 `castRay()` 中返回 -1 表示示没有任何东西被击中。然后我们在 `render()` 中处理。

```c++
vec3 render(vec3 rayOrigin, vec3 rayDir)
{
    vec3 col;
    // t stores the distance the ray travelled before intersecting a surface
    float t = castRay(rayOrigin, rayDir);
 
    // -1 means the ray didn't intersect anything, so render the skybox
    if (t == -1.0)
    {
        // Skybox colour
        col = vec3(0.30, 0.36, 0.60) - (rayDir.y * 0.7);
    }
    else
    {
        vec3 objectSurfaceColour = vec3(0.4, 0.8, 0.1);
        vec3 ambient = vec3(0.02, 0.021, 0.02);
        col = ambient * objectSurfaceColour;
    }
     
    return col;
}
```

### Diffuse term
为了得到更真实的光照，需要算一下表面法线，这样我们就可以计算出基本的 Lambertian 光照。为了计算法线，我们要计算表面在所有三个轴上的斜度。这意味着在实例中要对 SDF 进行四次额外的采样，每次采样都要与我们的主射线稍有偏移。
```c++
vec3 calcNormal(vec3 pos)
{
    // Center sample
    float c = sdf(pos);
    // Use offset samples to compute gradient / normal
    vec2 eps_zero = vec2(0.001, 0.0);
    return normalize(vec3( sdf(pos + eps_zero.xyy), sdf(pos + eps_zero.yxy), sdf(pos + eps_zero.yyx) ) - c);
}
```

检查法线计算好办法是把它们当作颜色来显示。这就是一个球体在显示其缩放和偏置法线时应该有的样子（从 `[-1, 1]` 变成 `[0, 1]` ，因为你的显示器不能显示负的颜色值）。
```C++
col = N * vec3(0.5) + vec3(0.5);
```

现在我们有了一个法向量，我们可以在它和光线方向之间取点积。这将告诉我们这个表面是如何直接面对光线的，这个点应该有多亮。我们取这个值的最大值为0，以防止负值给物体的暗面带来不必要的影响。

```C++
// L是从平面点到光源的的向量，N是平面法向量。N 和 L 都需要先 normalized
float NoL = max(dot(N, L), 0.0);
vec3 LDirectional = vec3(0.9, 0.9, 0.8) * NoL;
vec3 LAmbient = vec3(0.03, 0.04, 0.1);
vec3 diffuse = col * (LDirectional + LAmbient);
```

最后我们做一轮伽玛校正，这是一个很容易被忽视重要部分。发送到显示器的像素值是在伽马空间中的，这是一个非线性空间，伽马空间在人眼不太敏感的强度范围内使用较少的 bits，来最大限度地提高精度。

但由于显示器不是在 "线性 "空间中运行，我们需要在输出颜色之前对其伽玛曲线进行补偿。这种差异是非常明显的。在现实中，我们不知道特定显示设备的伽玛曲线是什么，所以显示技术的整个情况是非常混乱的（因此在许多游戏中都有伽玛调整步骤），比较常见的是以下伽玛曲线。
![](assets/Raymarching%20学习笔记/image-20221018145954370.png)
```c++
// The constant 0.4545 is simply 1.0 / 2.2
col = pow(col, vec3(0.4545)); // Gamma correction
```

![](assets/Raymarching%20学习笔记/image-20221018150325937.png)

原作者给出的阶段性源码：[https://www.shadertoy.com/view/4t3fzn](https://www.shadertoy.com/view/4t3fzn)

### Shadows
计算阴影阶段和传统光栅化不同，我们发射一条射线，从我们与场景表面相交的地方开始，**沿着光源的方向前进**。如果这条射线行进的结果是我们撞到了什么东西，那么我们就知道光线也会被阻挡，所以这个像素就处于阴影之中。
```c++
float shadow = 0.0;
vec3 shadowRayOrigin = pos + N * 0.01;
vec3 shadowRayDir = L;
IntersectionResult shadowRayIntersection = castRay(shadowRayOrigin, shadowRayDir);
if (shadowRayIntersection.mat != -1.0)
{
    shadow = 1.0;
}
col = mix(col, col*0.2, shadow);
```

### Ground plane
为了更好地看到球体投下的阴影，我们添加一个地平面
```C++
// p: plane origin (position), n.xyz: plane surface normal, p.w: plane's distance from origin (along its normal)
float sdPlane(vec3 p, vec4 n)
{
    return dot(p, n.xyz) + n.w;
}
```

### Soft shadows
现实生活中的阴影不会断崖式的消失，它会有一些衰减，被称为 penumbra。

我们可以从我们的表面点开始行进几条随机方向的射线来建立模型，每条射线的方向都略有不同。然后，我们将结果相加，并对所做的迭代次数进行平均。这将导致阴影的边缘有一些光线被击中，而另一些则没有被击中，从而产生50%的暗度。

寻找伪随机数的方法有很多，我们将使用以下方法：
```C++
// Return a psuedo random value in the range [0, 1), seeded via coord
float rand(vec2 coord)
{
  return fract(sin(dot(coord.xy, vec2(12.9898,78.233))) * 43758.5453);
}
```

最外层的操作是fract，它返回一个浮点数的小数部分，这个函数将返回一个范围为 `[0, 1]` 的数字，然后我们可以用它来计算我们的阴影射线：
```C++
float shadow = 0.0;
float shadowRayCount = 1.0;
for (float s = 0.0; s < shadowRayCount; s++)
{
    vec3 shadowRayOrigin = pos + N * 0.01;
    float r = rand(vec2(rayDir.xy)) * 2.0 - 1.0;
    vec3 shadowRayDir = L + vec3(1.0 * SHADOW_FALLOFF) * r;
    IntersectionResult shadowRayIntersection = castRay(shadowRayOrigin, shadowRayDir);
    if (shadowRayIntersection.mat != -1.0)
    {
        shadow += 1.0;
    }
}
col = mix(col, col*0.2, shadow/shadowRayCount);
```

## Texture mapping
与其在整个表面上统一定义单一的表面颜色，不如用纹理来定义应用于表面的图案。下面将介绍三种实现方法。

### 3D Texture mapping
在shadertoy中，有一些体积纹理可以被分配给一个通道。试着用表面点的三维位置对这些纹理进行采样。iChannel0中选取一个默认纹理。
```C++
// assign a 3D noise texture to iChannel0 and then sample based on world position
float textureFreq = 0.5;
vec3 surfaceCol = texture(iChannel0, textureFreq * surfacePos).xyz;
```
对噪声进行采样的一种方法是将多个标度加在一起，使用类似以下的方法。
```C++
// assign a 3D noise texture to iChannel0 and then sample based on world position
float textureFreq = 0.5;
vec3 surfaceCol =
    0.5    * texture(iChannel0, 1.0 * textureFreq * surfacePos).xyz +
    0.25   * texture(iChannel0, 2.0 * textureFreq * surfacePos).xyz +
    0.125  * texture(iChannel0, 4.0 * textureFreq * surfacePos).xyz +
    0.0625 * texture(iChannel0, 8.0 * textureFreq * surfacePos).xyz ;
```

### 2D Texture mapping
如何将纹理投射到表面？

传统的的三维管线中，物体每个三角形都有一个或多个uv，它们提供了纹理区域的坐标，应该映射到三角形上（纹理映射）。但我们的raymarching场景里是没有提供uv的。

一种方法是使用自上而下的世界投影对纹理进行采样，根据X和Z坐标对纹理进行采样。
```C++
// top down projection
float textureFreq = 0.5;
vec2 uv = textureFreq * surfacePos.xz;
 
// sample texture
vec3 surfaceCol = texture2D(iChannel0, uv).xyz;
```
这种方法有什么局限性？

### Triplanar mapping
一个更高级的贴图方法是在主轴上做3个投影，然后用三平面贴图混合结果。混合的目的是为表面上的每个点挑选最好的纹理。一种可能性是根据表面法线与每个世界轴的对齐程度来定义混合权重。朝向其中一个轴的表面将得到此轴最大的混合权重。
```C++
vec3 triplanarMap(vec3 surfacePos, vec3 normal)
{
    // Take projections along 3 axes, sample texture values from each projection, and stack into a matrix
    mat3 triMapSamples = mat3(
        texture(iChannel0, surfacePos.yz).rgb,
        texture(iChannel0, surfacePos.xz).rgb,
        texture(iChannel0, surfacePos.xy).rgb
        );
 
    // Weight three samples by absolute value of normal components
    return triMapSamples * abs(normal);
}
```
这种方法又有什么局限性？

### Materials
除了从castRay函数中返回的距离，我们还可以返回一个索引，代表被击中的物体的材料。我们可以使用这个索引来给物体涂上相应的颜色。我们的运算符将需要使用vec2，并比较每个运算符的第一个分量。

现在，在定义我们的场景时，我们还将为每个基元指定一个材料，作为vec2的y分量：
```C++
vec2 res =     vec2(sdSphere(pos-vec3(3,-2.5,10), 2.5),      0.1);
res = opU(res, vec2(sdSphere(pos-vec3(-3, -2.5, 10), 2.5),   2.0));
res = opU(res, vec2(sdSphere(pos-vec3(0, 2.5, 10), 2.5),     5.0));
return res;
```
这要求操作函数接受vec2而不是浮点数。下面是新版本的 union 运算符：
```C++
vec2 opU(vec2 d1, vec2 d2)
{
    return (d1.x < d2.x) ? d1 : d2;
}
```
新版本的castRay一直在跟踪最近的物体的材质ID，这样当我们决定击中一个表面时，我们就可以返回它的材质ID：
```C++
// Returns a vec2, x: signed distance to surface, y: material ID
vec2 castRay(vec3 rayOrigin, vec3 rayDir)
{
    float tmax = 250.0;
    // t stores the distance the ray travelled before intersecting a surface
    float t = 0.0;
    
    vec2 result;
    result.y = -1.0; // Default material ID
    
    for (int i = 0; i < 256; i++)
    {
        vec2 res = SDF(rayOrigin + rayDir * t);
        if (res.x < (0.0001*t))
        {
            // When within a small distance of the surface, count it as an intersection
            result.x = t;
            return result;
        }
        else if (res.x > tmax)
        {
            // Indicate that this ray didn't intersect anything
            result.y = -1.0;
            result.x = -1.0;
            return result;
        }
        t += res.x;
        result.y = res.y; // Material ID of closest object
    }
    
    result.x = t; // Distance to intersection
    return result;
}
```
我们的渲染函数可以改变以提取现在从castRay返回的两个字段：
```C++
vec2 res = castRay(rayOrigin, rayDir);
float t = res.x; // Distance to surface
float m = res.y; // Material ID
```
然后，我们可以在渲染函数中用这个材料指数乘以一些数值，为每个物体得到不同的颜色：
```C++
// m just stores some material identifier, here we're arbitrarily modifying it
// just to get some different colour values per object
col = vec3(0.18*m, 0.6-0.05*m, 0.2)
if (m == 2.0)
{
  // Apply triplanar mapping only to objects with material ID 2
    col *= triplanarMap(pos, N, 0.6);
}
```
让我们用一个棋盘格图案给地平面上色。我从Inigo Quilez的网站上下载了这个花哨的反锯齿的棋盘函数。
```C++
float checkers(vec2 p)
{
    vec2 w = fwidth(p) + 0.001;
    vec2 i = 2.0*(abs(fract((p-0.5*w)*0.5)-0.5)-abs(fract((p+0.5*w)*0.5)-0.5))/w;
    return 0.5 - 0.5*i.x*i.y;
}
```
我们将传入我们的平面位置的xz分量，以使图案在这些维度上重复。

最后根据每个交叉点发生在离摄像机多远的地方向场景中添加雾，减少迭代次数，你可以得到这样的效果：
![](assets/Raymarching%20学习笔记/image-20221018160015971.png)


### Blend
物体表面和物体材质的混合需要避免 min 运算符导致的折痕，我们可以使用一个更复杂的 sminCubic 运算符，将形状平滑地混合。
```C++
// polynomial smooth min (k = 0.1);
float sminCubic(float a, float b, float k)
{
    float h = max(k-abs(a-b), 0.0);
    return min(a, b) - h*h*h/(6.0*k*k);
}
 
vec2 opBlend(vec2 d1, vec2 d2)
{
    float k = 2.0;
    float d = sminCubic(d1.x, d2.x, k);
    float m = mix(d1.y, d2.y, clamp(d1.x-d,0.0,1.0));
    return vec2(d, m);
}
```
原作者例程：
![](assets/Raymarching%20学习笔记/image-20221018160202059.png)

### Anti-aliasing
通过对场景进行多次采样，并略微偏移摄像机方向的矢量，我们可以得到一个平滑的值，从而避免了混叠。我已经把场景颜色的计算带出了它自己的函数，以使在循环中调用它时更加清晰。
```C++
float AA_size = 2.0;
float count = 0.0;
for (float aaY = 0.0; aaY < AA_size; aaY++)
{
    for (float aaX = 0.0; aaX < AA_size; aaX++)
    {
        fragColor += getSceneColor(fragCoord + vec2(aaX, aaY) / AA_size);
        count += 1.0;
    }
}
fragColor /= count;
```

### Step count optimization
步数优化

如果我们把我们对每个像素所采取的步骤以红色显示出来，我们可以清楚地看到，那些什么也没打到的射线负责了我们大部分的迭代工作。部分场景能通过这种方式带来显著的性能提升。
```C++
if (t > drawDist) return backgroundColor;
```
![](assets/Raymarching%20学习笔记/image-20221018160812188.png)
![](assets/Raymarching%20学习笔记/image-20221018160912809.png)

### Shape and material interpolation
我们可以使用混合函数在两个形状之间进行插值，并使用iTime随时间调制。
```C++
vec2 shapeA = vec2(sdBox(pos-vec3(6.5, -3.0, 8), vec3(1.5)), 1.5);
vec2 shapeB = vec2(sdSphere(pos-vec3(6.5, -3.0, 8), 1.5),    3.0);
res = opU(res, mix(shapeA, shapeB, sin(iTime)*0.5+0.5));
```

### Domain repetition
使用 SDF 来重复绘制多个相同形状是非常容易的，基本上你只需要在一个或多个维度上对输入位置进行调制。例如，多次重复绘制球体，而不增加场景的表示尺寸。

这里我重复了输入位置的所有三个分量，然后使用减法运算符（max()）将重复限制在一个边界框内。
![](assets/Raymarching%20学习笔记/image-20221018161516906.png)

一个问题是，你需要减去你所调制的数值的一半，以便在你的形状上进行重复，而不是把它切成两半。

```C++
float repeat(float d, float domain)
{
    return mod(d, domain)-domain/2.0;
}
```

## Post processing effects
raymarching下的后处理效果

### Vignette

通过使离屏幕中心较远的像素变暗，我们可以得到一个简单的晕染效果。

### Contrast

较暗和较亮的数值可以被强调，导致感知的动态范围随着图像的强度而增加。
```C++
col = smoothstep(0.0,1.0,col);
```

### Ambient occlusio

如果我们对上图（在优化中）进行反演，我们可以得到一个类似AO的奇怪效果。
```C++
col *= (1.0-vec3(steps/maxSteps));
```
如你所见，许多后期处理效果可以通过各种琐碎的方式实现；玩玩不同的功能，看看你还能创造什么其他效果。

原作者例程：[www.shadertoy.com/view/MtdBzs](https://www.shadertoy.com/view/MtdBzs)

## What's next?
我们在这里只是介绍了基础知识；在这个领域还有很多东西需要探索：
- 次表层散射
- 环境遮蔽
- 动画基元
- 基元扭曲功能（扭曲、弯曲...）
- 透明度（折光、苛化......）
- 优化（边界体积层次）

### Recommended reading:

SDF functions: [http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/](http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/)
Claybook demo: [https://www.youtube.com/watch?v=Xpf7Ua3UqOA](https://www.youtube.com/watch?v=Xpf7Ua3UqOA)
Ray Tracing in One Weekend: [http://in1weekend.blogspot.com/2016/01/ray-tracing-in-one-weekend.html](http://in1weekend.blogspot.com/2016/01/ray-tracing-in-one-weekend.html)
Physically-based rendering bible, PBRT: [https://www.pbrt.org/](https://www.pbrt.org/)
Primitives reference: [http://iquilezles.org/www/articles/distfunctions/distfunctions.htm](http://iquilezles.org/www/articles/distfunctions/distfunctions.htm)
Extended introduction to building scenes with SDFs: [https://www.youtube.com/watch?v=s8nFqwOho-s](https://www.youtube.com/watch?v=s8nFqwOho-s)
Very realistic lighting & colours: [http://www.iquilezles.org/www/articles/outdoorslighting/outdoorslighting.htm](http://www.iquilezles.org/www/articles/outdoorslighting/outdoorslighting.htm)