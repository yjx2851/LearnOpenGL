#version 330 core
out vec4 FragColor;
in vec2 TexCoords;
in vec3 WorldPos;
in vec3 Normal;

// material parameters
uniform sampler2D albedoMap;
uniform sampler2D normalMap;
uniform sampler2D metallicMap;
uniform sampler2D roughnessMap;
uniform sampler2D aoMap;

// lights
uniform vec3 lightPositions[4];
uniform vec3 lightColors[4];

uniform vec3 camPos;

// subsurface scattering parameters
uniform float scatterDistance1;
uniform float scatterDistance2;
uniform float scatterWeight1;
uniform float scatterWeight2;
uniform float scatterPower;

const float PI = 3.14159265359;
// ----------------------------------------------------------------------------
// Easy trick to get tangent-normals to world-space to keep PBR code simplified.
// Don't worry if you don't get what's going on; you generally want to do normal 
// mapping the usual way for performance anyways; I do plan make a note of this 
// technique somewhere later in the normal mapping tutorial.
vec3 getNormalFromMap()
{
    vec3 tangentNormal = texture(normalMap, TexCoords).xyz * 2.0 - 1.0;

    vec3 Q1  = dFdx(WorldPos);
    vec3 Q2  = dFdy(WorldPos);
    vec2 st1 = dFdx(TexCoords);
    vec2 st2 = dFdy(TexCoords);

    vec3 N   = normalize(Normal);
    vec3 T  = normalize(Q1*st2.t - Q2*st1.t);
    vec3 B  = -normalize(cross(N, T));
    mat3 TBN = mat3(T, B, N);

    return normalize(TBN * tangentNormal);
}
// ----------------------------------------------------------------------------
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

// ----------------------------------------------------------------------------
float GeneralizedTrowbridgeReitz(float c,vec3 N, vec3 H, float roughness,float gamma)
{
    float a = roughness*roughness;
    float NdotH2=max(dot(N,H),0.0)*max(dot(N,H),0.0);
    float denom=pow(a*NdotH2+(1-NdotH2),gamma);
    return c/denom;
}

// ----------------------------------------------------------------------------
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}
// ----------------------------------------------------------------------------
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ----------------------------------------------------------------------------
// 次表面散射：使用两个指数项的总和来模拟光在材质内部的散射行为
vec3 SubsurfaceScattering(vec3 L, vec3 V, vec3 N, vec3 lightColor, float thickness)
{
    // 计算入射角对透射的影响（Fresnel透射效应）
    float NdotL = max(dot(N, L), 0.0);
    float transmissionFactor = 1.0 - pow(1.0 - NdotL, 5.0); // 简化的透射Fresnel
    
    // 计算透射光方向 (光线经过材质后的方向)
    vec3 H = normalize(L + N * scatterPower);
    
    // 计算光线与法线的夹角，用于计算散射强度
    float VdotH = max(dot(-V, H), 0.0);
    float LdotH = max(dot(L, H), 0.0);
    
    // 使用两个指数项的总和来模拟不同深度的散射
    // 第一个指数项：浅层散射，主要影响近表面的散射
    float scatter1 = exp(-thickness / scatterDistance1);
    // 第二个指数项：深层散射，模拟更深层的光线传播
    float scatter2 = exp(-thickness / scatterDistance2);
    
    // 组合两个散射项，使用不同的权重
    float totalScatter = scatterWeight1 * scatter1 + scatterWeight2 * scatter2;
    
    // 考虑视角和光线方向的影响
    float scatterProfile = pow(max(VdotH, 0.0), scatterPower) * max(LdotH, 0.0);
    
    // 计算最终的次表面散射贡献
    // 应用透射因子来模拟入射角的影响
    vec3 subsurface = lightColor * totalScatter * scatterProfile * transmissionFactor;
    
    return subsurface;
}

// ----------------------------------------------------------------------------
void main()
{		
    vec3 albedo     = pow(texture(albedoMap, TexCoords).rgb, vec3(2.2));
    float metallic  = texture(metallicMap, TexCoords).r;
    float roughness = texture(roughnessMap, TexCoords).r;
    float ao        = texture(aoMap, TexCoords).r;

    vec3 N = getNormalFromMap();
    vec3 V = normalize(camPos - WorldPos);

    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
    // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)    
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);

    // reflectance equation
    vec3 Lo = vec3(0.0);
    for(int i = 0; i < 4; ++i) 
    {
        // calculate per-light radiance
        vec3 L = normalize(lightPositions[i] - WorldPos);
        vec3 H = normalize(V + L);
        float distance = length(lightPositions[i] - WorldPos);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = lightColors[i] * attenuation;

        // Cook-Torrance BRDF
        // float NDF = DistributionGGX(N, H, roughness);   
        // 主镜面 GTR分布
        float specular_GTR = GeneralizedTrowbridgeReitz(roughness*roughness/PI,N, H, roughness, 2.0);

        // 清漆层 GTR分布
        float varnish_GTR = GeneralizedTrowbridgeReitz((roughness*roughness)/(PI*log(roughness*roughness + 0.01)),N,H,roughness, 1.0);
        float GTR = specular_GTR + varnish_GTR;
        
        float G   = GeometrySmith(N, V, L, roughness);      
        vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);
           
        vec3 numerator    = specular_GTR * G * F; 
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // + 0.0001 to prevent divide by zero
        vec3 specular = numerator / denominator;
        
        // kS is equal to Fresnel
        vec3 kS = F;
        // for energy conservation, the diffuse and specular light can't
        // be above 1.0 (unless the surface emits light); to preserve this
        // relationship the diffuse component (kD) should equal 1.0 - kS.
        vec3 kD = vec3(1.0) - kS;
        // multiply kD by the inverse metalness such that only non-metals 
        // have diffuse lighting, or a linear blend if partly metal (pure metals
        // have no diffuse light).
        kD *= 1.0 - metallic;	  

        // scale light by NdotL
        float NdotL = max(dot(N, L), 0.0);        

        // 次表面散射计算 (适用于非金属材质)
        // 使用roughness纹理的绿色通道作为材质厚度信息
        float thickness = texture(roughnessMap, TexCoords).g;
        vec3 subsurface = SubsurfaceScattering(L, V, N, radiance, thickness);
        
        // 次表面散射主要影响非金属材质
        subsurface *= (1.0 - metallic) * albedo;

        // add to outgoing radiance Lo
        Lo += (kD * albedo / PI + specular) * radiance * NdotL + subsurface;  // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again
    }   
    
    // ambient lighting (note that the next IBL tutorial will replace 
    // this ambient lighting with environment lighting).
    vec3 ambient = vec3(0.03) * albedo * ao;
    
    vec3 color = ambient + Lo;

    // HDR tonemapping
    color = color / (color + vec3(1.0));
    // gamma correct
    color = pow(color, vec3(1.0/2.2)); 

    FragColor = vec4(color, 1.0);
}