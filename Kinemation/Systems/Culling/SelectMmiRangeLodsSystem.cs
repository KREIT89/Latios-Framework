using Latios.Transforms;
using Latios.Transforms.Abstract;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Entities;
using Unity.Entities.Exposed;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Rendering;

using static Unity.Entities.SystemAPI;

namespace Latios.Kinemation.Systems
{
    [DontSyncPreviousUpdatesThisFrame(32)]
    [RequireMatchingQueriesForUpdate]
    [DisableAutoCreation]
    [BurstCompile]
    public partial struct SelectMmiRangeLodsSystem : ISystem, ISystemShouldUpdate
    {
        LatiosWorldUnmanaged                    latiosWorld;
        WorldTransformReadOnlyAspect.TypeHandle m_worldTransformHandle;

        EntityQuery m_query;

        int   m_maximumLODLevel;
        float m_lodBias;
        float m_meshLodThreshold;

        // KREIT89 fork (LodTimedFade): captured on the main thread in ShouldUpdateSystem, used in OnUpdate.
        float3 m_mainCameraPosition;
        bool   m_hasMainCamera;
        float  m_unscaledDeltaTime;
        int    m_frameCount;
        int    m_lastFadeFrame;

        [BurstCompile]
        public void OnCreate(ref SystemState state)
        {
            latiosWorld            = state.GetLatiosWorldUnmanaged();
            m_worldTransformHandle = new WorldTransformReadOnlyAspect.TypeHandle(ref state);

            m_query = state.Fluent().With<MaterialMeshInfo, LodCrossfade>(false).With<WorldRenderBounds>(true)
                      .WithAnyEnabled<MmiRange2LodSelect, MmiRange3LodSelect, MeshLodCurve>(true).WithWorldTransformReadOnly().Build();

            latiosWorld.worldBlackboardEntity.AddComponentDataIfMissing(new MeshLodCrossfadeMargin { margin = (half)0.05f });
            latiosWorld.worldBlackboardEntity.AddComponentDataIfMissing(new LodTimedFadeSettings { fadeSeconds = 0.5f, hysteresis = 0.05f });
            m_lastFadeFrame = -1;
        }

        public bool ShouldUpdateSystem(ref SystemState state)
        {
            m_maximumLODLevel = UnityEngine.QualitySettings.maximumLODLevel;
            m_lodBias         = UnityEngine.QualitySettings.lodBias;
            // LodTimedFade: which camera advances fades (the main one, not the Editor scene view), and by how much.
            var mainCam          = UnityEngine.Camera.main;
            m_hasMainCamera      = mainCam != null;
            m_mainCameraPosition = m_hasMainCamera ? (float3)mainCam.transform.position : float3.zero;
            m_unscaledDeltaTime  = UnityEngine.Time.unscaledDeltaTime;
            m_frameCount         = UnityEngine.Time.frameCount;
#if UNITY_6000_2_OR_NEWER
            m_meshLodThreshold = UnityEngine.QualitySettings.meshLodThreshold;
#endif
            return m_maximumLODLevel < 2;
        }

        [BurstCompile]
        public void OnUpdate(ref SystemState state)
        {
            var context    = latiosWorld.worldBlackboardEntity.GetComponentData<CullingContext>();
            var parameters = context.lodParameters;

            m_worldTransformHandle.Update(ref state);

            float cameraFactorNoBias = LodUtilities.CameraFactorFrom(in parameters, 1f);

            // LodTimedFade: advance only in the main camera's pass, and only once per frame. The camera is recognised by
            // position (the Editor scene view culls too, often first). Everything else reuses the stored state.
            bool isMainCameraPass = m_hasMainCamera && context.viewType == UnityEngine.Rendering.BatchCullingViewType.Camera &&
                                    math.distancesq(parameters.cameraPosition, m_mainCameraPosition) < 1e-4f;
            bool advanceFades = isMainCameraPass && m_frameCount != m_lastFadeFrame;
            if (advanceFades)
                m_lastFadeFrame = m_frameCount;
            var fadeSettings = latiosWorld.worldBlackboardEntity.GetComponentData<LodTimedFadeSettings>();

            state.Dependency = new Job
            {
                perCameraMaskHandle    = GetComponentTypeHandle<ChunkPerCameraCullingMask>(false),
                worldTransformHandle   = m_worldTransformHandle,
                boundsHandle           = GetComponentTypeHandle<WorldRenderBounds>(true),
                select2Handle          = GetComponentTypeHandle<MmiRange2LodSelect>(true),
                select3Handle          = GetComponentTypeHandle<MmiRange3LodSelect>(true),
                rangeLodFlagsHandle    = GetComponentTypeHandle<MmiRangeLodFlags>(true),
                lodGroupCrossfades     = GetComponentTypeHandle<LodHeightPercentagesWithCrossfadeMargins>(true),
                mmiHandle              = GetComponentTypeHandle<MaterialMeshInfo>(false),
                crossfadeHandle        = GetComponentTypeHandle<LodCrossfade>(false),
                meshLodHandle          = GetComponentTypeHandle<MeshLod>(false),
                meshLodCurveHandle     = GetComponentTypeHandle<MeshLodCurve>(true),
                timedFadeHandle        = GetComponentTypeHandle<LodTimedFade>(false),
                advanceFades           = advanceFades,
                fadeStep               = fadeSettings.fadeSeconds > 0f ? m_unscaledDeltaTime / fadeSettings.fadeSeconds : 1f,
                fadeHysteresis         = math.clamp(fadeSettings.hysteresis, 0f, 0.5f),
                frame16                = (ushort)math.max(1, m_frameCount & 0xffff),
                meshLodCrossfadeMargin = latiosWorld.worldBlackboardEntity.GetComponentData<MeshLodCrossfadeMargin>().margin,
                cameraPosition         = parameters.cameraPosition,
                isPerspective          = !parameters.isOrthographic,
                isShadowCasting        = context.viewType == UnityEngine.Rendering.BatchCullingViewType.Light,
                cameraFactor           = cameraFactorNoBias * m_lodBias,
                meshLodFactor          = m_meshLodThreshold / (cameraFactorNoBias * parameters.cameraPixelHeight),
                inverseLodBias         = 1f / m_lodBias,
                maxResolutionLodLevel  = m_maximumLODLevel
            }.ScheduleParallel(m_query, state.Dependency);
        }

        [BurstCompile]
        unsafe struct Job : IJobChunk
        {
            [ReadOnly] public WorldTransformReadOnlyAspect.TypeHandle                       worldTransformHandle;
            [ReadOnly] public ComponentTypeHandle<WorldRenderBounds>                        boundsHandle;
            [ReadOnly] public ComponentTypeHandle<MmiRange2LodSelect>                       select2Handle;
            [ReadOnly] public ComponentTypeHandle<MmiRange3LodSelect>                       select3Handle;
            [ReadOnly] public ComponentTypeHandle<MmiRangeLodFlags>                         rangeLodFlagsHandle;
            [ReadOnly] public ComponentTypeHandle<LodHeightPercentagesWithCrossfadeMargins> lodGroupCrossfades;
            [ReadOnly] public ComponentTypeHandle<MeshLodCurve>                             meshLodCurveHandle;

            public ComponentTypeHandle<ChunkPerCameraCullingMask> perCameraMaskHandle;
            public ComponentTypeHandle<MaterialMeshInfo>          mmiHandle;
            public ComponentTypeHandle<LodCrossfade>              crossfadeHandle;
            public ComponentTypeHandle<MeshLod>                   meshLodHandle;
            public ComponentTypeHandle<LodTimedFade>              timedFadeHandle;
            public bool                                           advanceFades;
            public float                                          fadeStep;
            public float                                          fadeHysteresis;
            public ushort                                         frame16;

            public float3 cameraPosition;
            public float  cameraFactor;
            public float  meshLodFactor;
            public float  inverseLodBias;
            public float  meshLodCrossfadeMargin;
            public int    maxResolutionLodLevel;
            public bool   isPerspective;
            public bool   isShadowCasting;

            public void Execute(in ArchetypeChunk chunk, int unfilteredChunkIndex, bool useEnabledMask, in v128 chunkEnabledMask)
            {
                ref var mask = ref chunk.GetChunkComponentRefRW(ref perCameraMaskHandle);
                if ((mask.upper.Value | mask.lower.Value) == 0)
                    return;

                var transforms              = worldTransformHandle.Resolve(chunk);
                var boundsArray             = (WorldRenderBounds*)chunk.GetRequiredComponentDataPtrRO(ref boundsHandle);
                var mmis                    = (MaterialMeshInfo*)chunk.GetRequiredComponentDataPtrRW(ref mmiHandle);
                var crossfades              = (LodCrossfade*)chunk.GetRequiredComponentDataPtrRW(ref crossfadeHandle);
                var crossfadesEnabled       = chunk.GetEnabledMask(ref crossfadeHandle);
                var select2s                = chunk.GetComponentDataPtrRO(ref select2Handle);
                var select3s                = chunk.GetComponentDataPtrRO(ref select3Handle);
                var rangeLodFlagsArray      = isShadowCasting ? chunk.GetComponentDataPtrRO(ref rangeLodFlagsHandle) : null;
                var lodGroupPercentages     = chunk.GetComponentDataPtrRO(ref lodGroupCrossfades);
                var meshLods                = chunk.GetComponentDataPtrRW(ref meshLodHandle);
                var enableMeshLodCrossfades = chunk.GetEnabledMask(ref meshLodHandle);
                var meshLodCurves           = chunk.GetComponentDataPtrRO(ref meshLodCurveHandle);
                var timedFades              = (LodTimedFade*)chunk.GetComponentDataPtrRW(ref timedFadeHandle);
                var enumerator              = new ChunkEntityEnumerator(true, new v128(mask.lower.Value, mask.upper.Value), chunk.Count);
                while (enumerator.NextEntityIndex(out int i))
                {
                    MmiRange3LodSelect select;
                    bool3              nullSelect;
                    MaterialMeshInfo   mmi;
                    int                maxLodSupported;
                    if (select3s != null)
                    {
                        select                                 = select3s[i];
                        nullSelect                             = new bool3(false, select.fullLod1ScreenHeightMaxFraction < 0f, select.fullLod2ScreenHeightFraction < 0f);
                        select.fullLod1ScreenHeightMaxFraction = (half)math.abs(select.fullLod1ScreenHeightMaxFraction);
                        select.fullLod2ScreenHeightFraction    = (half)math.abs(select.fullLod2ScreenHeightFraction);
                        mmi                                    = mmis[i];
                        maxLodSupported                        = 2;
                    }
                    else if (select2s != null)
                    {
                        select = new MmiRange3LodSelect
                        {
                            fullLod0ScreenHeightFraction    = select2s[i].fullLod0ScreenHeightFraction,
                            fullLod1ScreenHeightMaxFraction = (half)math.abs(select2s[i].fullLod1ScreenHeightFraction),
                            fullLod1ScreenHeightMinFraction = default,
                            fullLod2ScreenHeightFraction    = default,
                        };
                        nullSelect      = new bool3(false, new bool2(select2s[i].fullLod1ScreenHeightFraction < 0f));
                        mmi             = mmis[i];
                        maxLodSupported = 1;
                    }
                    else
                    {
                        select = new MmiRange3LodSelect
                        {
                            fullLod0ScreenHeightFraction    = default,
                            fullLod1ScreenHeightMaxFraction = default,
                            fullLod1ScreenHeightMinFraction = default,
                            fullLod2ScreenHeightFraction    = default,
                        };
                        nullSelect      = default;
                        mmi             = default;
                        maxLodSupported = 0;
                    }
                    if (rangeLodFlagsArray != null)
                    {
                        var flags     = rangeLodFlagsArray[i];
                        nullSelect.y &= !flags.disableLod1ShadowCasting;
                        nullSelect.z &= !flags.disableLod2ShadowCasting;
                    }

                    float  height = math.cmax(boundsArray[i].Value.Extents) * 2f;
                    float3 center = boundsArray[i].Value.Center;
                    float  groupMin, groupMax;
                    if (lodGroupPercentages != null)
                    {
                        // We need to convert group LOD thresholds into local LOD thresholds. The key differences is that they use different center points and relative heights.
                        var   transform        = transforms[i].worldTransformQvvs;
                        float groupWorldHeight = math.abs(lodGroupPercentages[i].localSpaceHeight) * math.abs(transform.scale) * math.cmax(math.abs(transform.stretch));
                        float factor           = height / groupWorldHeight;
                        if (isPerspective)
                            factor *= math.distance(cameraPosition, transform.position) / math.distance(cameraPosition, center);
                        groupMin    = factor * lodGroupPercentages[i].minCrossFadeEdge;
                        groupMax    = factor * lodGroupPercentages[i].maxCrossFadeEdge;
                    }
                    else
                    {
                        groupMin = 0f;
                        groupMax = float.MaxValue;
                    }

                    MeshLod      meshLodDummy = default;
                    ref var      meshLod      = ref meshLodDummy;
                    MeshLodCurve meshLodCurve = default;
                    if (meshLods != null && meshLodCurves != null)
                    {
                        meshLod      = ref meshLods[i];
                        meshLodCurve = meshLodCurves[i];
                    }

                    DoEntity(ref mmi,
                             ref crossfades[i],
                             out var crossfadeEnabled,
                             out var cull,
                             select,
                             nullSelect,
                             center,
                             height,
                             groupMin,
                             groupMax,
                             maxLodSupported,
                             ref meshLod,
                             meshLodCurve,
                             out var enableMeshLodCrossfade);

                    // KREIT89 fork: LodTimedFade overrides the LOD region and crossfade chosen above.
                    if (!cull && timedFades != null && lodGroupPercentages == null && maxLodSupported > 0 && !math.any(nullSelect.yz))
                    {
                        DoTimedFade(ref timedFades[i], ref mmi, ref crossfades[i], out crossfadeEnabled, select, center, height, maxLodSupported);
                    }

                    if (cull)
                        mask.ClearBitAtIndex(i);
                    if (lodGroupPercentages == null || !crossfadesEnabled[i])
                        crossfadesEnabled[i] = crossfadeEnabled;
                    if (enableMeshLodCrossfade)
                        enableMeshLodCrossfades[i] = true;
                    if (select2s != null || select3s != null)
                        mmis[i] = mmi;
                }
            }

            // KREIT89 fork (NethCoris 2026-09-30): time-based LOD crossfade. See LodTimedFade.
            void DoTimedFade(ref LodTimedFade tf, ref MaterialMeshInfo mmi, ref LodCrossfade crossfade, out bool crossfadeEnabled,
                             MmiRange3LodSelect select, float3 center, float height, int maxLodSupported)
            {
                var biasHeight = height * cameraFactor;
                var distance   = math.select(1f, math.distance(center, cameraPosition), isPerspective);
                var h01        = select.fullLod0ScreenHeightFraction * distance;      // LOD0 <-> LOD1 switch
                var h12        = select.fullLod1ScreenHeightMinFraction * distance;   // LOD1 <-> LOD2 switch (3-level only)
                int minLevel   = math.min(maxResolutionLodLevel, maxLodSupported);

                if (advanceFades)
                {
                    // Level with the thresholds pushed out (coarsest allowed) and pulled in (finest allowed).
                    int coarse = LevelAt(biasHeight, h01 * (1f + fadeHysteresis), h12 * (1f + fadeHysteresis), maxLodSupported);
                    int fine   = LevelAt(biasHeight, h01 * (1f - fadeHysteresis), h12 * (1f - fadeHysteresis), maxLodSupported);
                    coarse     = math.max(coarse, minLevel);
                    fine       = math.max(fine, minLevel);

                    ushort previous = tf.lastFrame;
                    bool   stale    = previous == 0 || (ushort)(frame16 - previous) > 2;   // not seen recently: snap, nobody watched
                    tf.lastFrame    = frame16;

                    if (stale)
                    {
                        int plain     = math.max(LevelAt(biasHeight, h01, h12, maxLodSupported), minLevel);
                        tf.currentLod = (byte)plain;
                        tf.targetLod  = (byte)plain;
                        tf.progress   = default;
                    }
                    else
                    {
                        int cur = tf.currentLod;
                        // The level we want: stay unless outside the margin band.
                        int wanted = cur < fine ? fine : (cur > coarse ? coarse : cur);
                        float p = tf.progress;
                        if (tf.targetLod == cur)
                        {
                            if (wanted != cur)
                            {
                                tf.targetLod = (byte)(cur + (wanted > cur ? 1 : -1));   // adjacent levels only
                                p            = math.min(1f, fadeStep);
                            }
                        }
                        else
                        {
                            int  dir     = tf.targetLod > cur ? 1 : -1;
                            bool towards = (wanted - cur) * dir > 0;
                            p            = towards ? p + fadeStep : p - fadeStep;   // reverses smoothly, never snaps back
                        }
                        if (p >= 1f)
                        {
                            tf.currentLod = tf.targetLod;
                            p             = 0f;
                        }
                        else if (p <= 0f && tf.targetLod != tf.currentLod)
                        {
                            tf.targetLod = tf.currentLod;
                            p            = 0f;
                        }
                        tf.progress = (half)p;
                    }
                }
                else if (tf.lastFrame == 0)
                {
                    // Never advanced yet (e.g. only seen by a shadow pass so far): show the plain level, don't store it.
                    int plain = math.max(LevelAt(biasHeight, h01, h12, maxLodSupported), minLevel);
                    crossfadeEnabled = false;
                    mmi.SetCurrentLodRegion(plain, false);
                    return;
                }

                if (tf.targetLod == tf.currentLod)
                {
                    crossfadeEnabled = false;
                    mmi.SetCurrentLodRegion(tf.currentLod, false);
                }
                else
                {
                    int   lo          = math.min((int)tf.currentLod, (int)tf.targetLod);   // byte args are ambiguous for math.min
                    float progress    = math.saturate((float)tf.progress);
                    float hiResOpacity = tf.currentLod == lo ? 1f - progress : progress;   // opacity of the finer level (lo)
                    crossfadeEnabled  = true;
                    mmi.SetCurrentLodRegion(lo, true);
                    crossfade.SetFromHiResOpacity(hiResOpacity, false);
                }
            }

            static int LevelAt(float biasHeight, float h01, float h12, int maxLodSupported)
            {
                if (biasHeight >= h01 || maxLodSupported < 1) return 0;
                if (maxLodSupported < 2 || biasHeight >= h12) return 1;
                return 2;
            }

            void DoEntity(ref MaterialMeshInfo mmi,
                          ref LodCrossfade crossfade,
                          out bool crossfadeEnabled,
                          out bool cull,
                          MmiRange3LodSelect select,
                          bool3 nullSelect,
                          float3 center,
                          float height,
                          float groupMin,
                          float groupMax,
                          int maxLodSupported,
                          ref MeshLod meshLod,
                          in MeshLodCurve meshLodCurve,
                          out bool enableMeshLodCrossfade)
            {
                cull                   = false;
                enableMeshLodCrossfade = false;
                int minLod             = 0;
                int maxLod             = 2;
                if (select.fullLod1ScreenHeightMinFraction < groupMin)
                    maxLod = 0;
                else if (select.fullLod2ScreenHeightFraction < groupMin)
                    maxLod = 1;
                if (select.fullLod1ScreenHeightMaxFraction > groupMax)
                    minLod = 2;
                else if (select.fullLod0ScreenHeightFraction > groupMax)
                    minLod = 1;
                minLod     = math.max(minLod, maxResolutionLodLevel);
                maxLod     = math.max(maxLod, maxResolutionLodLevel);
                minLod     = math.min(minLod, maxLodSupported);
                maxLod     = math.min(maxLod, maxLodSupported);
                groupMin   = math.min(groupMin, select.fullLod0ScreenHeightFraction);
                if (minLod == maxLod)
                {
                    crossfadeEnabled = false;
                    mmi.SetCurrentLodRegion(minLod, false);
                }
                else
                {
                    if (minLod == 1)
                    {
                        select.fullLod1ScreenHeightMaxFraction = half.MaxValueAsHalf;
                        select.fullLod0ScreenHeightFraction    = half.MaxValueAsHalf;
                    }
                    else if (maxLod == 1)
                    {
                        select.fullLod1ScreenHeightMinFraction = default;
                        select.fullLod2ScreenHeightFraction    = (half)math.min(select.fullLod2ScreenHeightFraction, 0f);
                    }

                    var biasHeight = height * cameraFactor;
                    var distance   = math.select(1f, math.distance(center, cameraPosition), isPerspective);

                    var zeroHeight   = select.fullLod0ScreenHeightFraction * distance;
                    var oneMaxHeight = select.fullLod1ScreenHeightMaxFraction * distance;
                    var oneMinHeight = select.fullLod1ScreenHeightMinFraction * distance;
                    var twoHeight    = select.fullLod2ScreenHeightFraction * distance;

                    if (biasHeight >= zeroHeight)
                    {
                        crossfadeEnabled = false;
                        mmi.SetCurrentLodRegion(0, false);
                    }
                    else if (biasHeight <= twoHeight)
                    {
                        crossfadeEnabled = false;
                        mmi.SetCurrentLodRegion(2, false);
                        if (nullSelect.z)
                            cull = true;
                    }
                    else if ((biasHeight <= oneMaxHeight) && biasHeight >= oneMinHeight)
                    {
                        crossfadeEnabled = false;
                        mmi.SetCurrentLodRegion(1, false);
                        if (nullSelect.y)
                            cull = true;
                    }
                    else if (biasHeight > oneMaxHeight)
                    {
                        crossfadeEnabled = true;
                        mmi.SetCurrentLodRegion(0, true);
                        crossfade.SetFromHiResOpacity(math.unlerp(oneMaxHeight, zeroHeight, biasHeight), false);
                    }
                    else
                    {
                        crossfadeEnabled = true;
                        mmi.SetCurrentLodRegion(1, true);
                        crossfade.SetFromHiResOpacity(math.unlerp(twoHeight, oneMinHeight, biasHeight), false);
                    }
                }

                if (meshLod.levelCount <= 0 || cull)
                    return;

                var heights            = new float3(height, groupMax, groupMin);
                heights.yz            *= inverseLodBias;
                var meshLodDistance    = math.select(1f, math.distance(center, cameraPosition), isPerspective);
                var screenFractions    = meshLodFactor * meshLodDistance / heights;
                var preClamp           = math.log2(screenFractions) * meshLodCurve.slope + meshLodCurve.preClampBias;
                var postClamp          = math.max(0f, preClamp) + meshLodCurve.postClampBias;
                var rounded            = math.round(postClamp);
                var isWithinCrossfade  = math.abs(postClamp - rounded) <= meshLodCrossfadeMargin;
                var groupClampRegion   = rounded + math.select(float3.zero, new float3(0f, meshLodCrossfadeMargin, -meshLodCrossfadeMargin), isWithinCrossfade);
                var meshLodLevel       = math.clamp(groupClampRegion.x, groupClampRegion.y, groupClampRegion.z);
                meshLodLevel           = math.min(meshLodLevel, meshLod.levelCount - 1.5f);  // The extra 0.5 is to prevent crossfading
                var meshLodRounded     = math.round(meshLodLevel);
                if (math.distance(meshLodLevel, meshLodRounded) < meshLodCrossfadeMargin)
                {
                    var hiResOpacity = math.unlerp(meshLodRounded - meshLodCrossfadeMargin, meshLodRounded + meshLodCrossfadeMargin, meshLodLevel);
                    meshLod.lodLevel = (ushort)meshLodLevel;
                    if (meshLodLevel > meshLodRounded)
                        meshLod.lodLevel--;
                    crossfade.SetFromHiResOpacity(hiResOpacity, false);
                    crossfadeEnabled       = true;
                    enableMeshLodCrossfade = true;
                }
                else
                {
                    crossfadeEnabled = false;
                    meshLod.lodLevel = (ushort)meshLodLevel;
                }
            }
        }
    }
}

