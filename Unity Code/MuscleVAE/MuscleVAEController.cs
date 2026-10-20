using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using Kinesis.Components;
using Kinesis.Core;
using Unity.MLAgents.Actuators;
using Unity.VisualScripting;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;
using static MuscleVAE.Utils.MotionUtils;
using Random = UnityEngine.Random;

[DefaultExecutionOrder(-200)]
public class MuscleVAEController : MonoBehaviour
{
    [Serializable]
    public class BoneMapEntry
    {
        public Transform stateBone;
        public Transform targetBone;
        public float weight = 1.0f;
    }

    [Serializable]
    public class MuscleMapEntry
    {
        public MuscleTendonUnit stateMuscle;
        public MuscleTendonUnit targetMuscle;
        public float weight = 1.0f;
    }
    
    public MuscleVAECharacter _base { get; private set; }
    public MuscleVAECharacter _target { get; private set; }

        
    [Header("Rigid Bodies")]
    public List<BoneMapEntry> bodies = new();
    
    [Header("Muscles")]
    public List<MuscleMapEntry> muscles = new();
    
    [Header("EndEffectors")]
    public List<BoneMapEntry> endEffectors = new();
    
    [Header("Roots")]
    public List<BoneMapEntry> root = new();
    
    [Header("Head")]
    public List<BoneMapEntry> baseHead = new();
    
    [Header("Animator")]
    public Animator animator;
    
    [Header("Muscles")]
    public MuscleGroup BaseMuscleGroup;
    public MuscleGroup TargetMuscleGroup;
    
    [Header("Keywords")]
    public string[] weightKeyWords = {"index", "middle", "ring", "pinky", "thumb"};
    public string[] eeKeyWords = {"hand", "foot", "head"};
    public string[] excludeKeyWords = {"collider", "cube", "toe_end", "4", "_end", "alpha"};
    public string[] headkeyWords = {"Head"};


    #region caches
    List<Transform> baseBones = new List<Transform>();
    List<Transform> targetBones = new List<Transform>();
    
    List<float> weights = new List<float>();
    
    List<MuscleTendonUnit> baseMuscles = new List<MuscleTendonUnit>();
    List<MuscleTendonUnit> targetMuscles = new List<MuscleTendonUnit>();
    
    private float animationPhase = 0f;
    #endregion
    
    private void Awake()
    {
        _base = new MuscleVAECharacter();
        _target = new MuscleVAECharacter();
        
        BuildCaches();
        _base.CountActions();
        Debug.Log($"Found {baseBones.Count} base bones, {targetBones.Count} target bones.");
        Debug.Log($"Found {baseMuscles.Count} base muscles, {targetMuscles.Count} target muscles.");
    }

    private void FixedUpdate()
    {
        UpdateCaches();
    }
    
    public void BuildCaches()
    {
        animationPhase = 0;
        baseBones.Clear();
        targetBones.Clear();
        weights.Clear();
        foreach (var b in bodies)
        {
            baseBones.Add(b.stateBone);
            targetBones.Add(b.targetBone);
            weights.Add(b.weight);
        }
        
        baseMuscles.Clear();
        targetMuscles.Clear();
        foreach (var m in muscles)
        {
            baseMuscles.Add(m.stateMuscle);
            targetMuscles.Add(m.targetMuscle);
        }
        
        
        //Debug.Log("Base muscles count: "+baseMuscles.Count);
        _base.BuildCaches(baseBones, baseMuscles ,root[0].stateBone, baseHead[0].stateBone);
        _target.BuildCaches(targetBones,targetMuscles ,root[0].targetBone, baseHead[0].targetBone);
        
        animationPhase = animator.GetCurrentAnimatorStateInfo(0).normalizedTime;
    }
    
    public void UpdateCaches()
    {
        _base.UpdateCaches(baseBones, baseMuscles ,root[0].stateBone, baseHead[0].stateBone);
        _target.UpdateCaches(targetBones, targetMuscles ,root[0].targetBone, baseHead[0].targetBone);
        
        animationPhase = animator.GetCurrentAnimatorStateInfo(0).normalizedTime;
    }

    public List<float> GetWeights()
    {
        List<float> w = new List<float>();
        foreach (var b in bodies)
        {
            w.Add(b.weight);
        }

        return w;
    }
    
    public void BuildWaypointArrays(out List<Vector3> offsets, out List<int> localIndices, out List<int> bodyIndices)
    {
        offsets = new List<Vector3>();
        localIndices = new List<int>();
        bodyIndices = new List<int>();

        // Build lookup once
        Dictionary<Transform, int> cacheIndex = new Dictionary<Transform, int>();
        for (int i = 0; i < bodies.Count; i++)
        {
            cacheIndex[bodies[i].targetBone] = i;
        }

        for (int i = 0; i < muscles.Count; i++)
        {
            var muscle = muscles[i].targetMuscle;

            for (int j = 0; j < muscle.muscleNodes.Count; j++)
            {
                var node = muscle.muscleNodes[j];

                // Offset
                offsets.Add(node.offset);

                // Local index (0,1,2,... per muscle)
                localIndices.Add(j);
                //Debug.Log(j);

                // Body index
                var body = node.bone.transform;

                if (!cacheIndex.TryGetValue(body, out int idx))
                    throw new Exception($"Body not found in cache: {body.name}");

                bodyIndices.Add(idx);
                //Debug.Log(idx);
            }
        }
        Debug.Log("Number of body indices: " + bodyIndices.Count);
        Debug.Log("Number of local indices: " + localIndices.Count);
        Debug.Log("Number of offsets: " + offsets.Count);
    }

    public void SetRandomTargets()
    {
        int size = muscles.Count;
        float[] values = new float[size];

        for (int i = 0; i < size; i++)
        {
            values[i] = UnityEngine.Random.value; //0 to 1
        }
        SetTargets(values);
    }
    public void SetTargets(ActionBuffers buffers)
    {
        List<float> actions = new List<float>();
        var acts = buffers.ContinuousActions;
        for (int i = 0; i < acts.Length; i++)
        {
            actions.Add(acts[i]);
        }
        _base.SetMuscleTargets(actions);
    }
    public void SetTargets(float[] buffers)
    {
        List<float> actions = new List<float>();
        var acts = buffers;
        for (int i = 0; i < acts.Length; i++)
        {
            actions.Add(acts[i]);
        }
        _base.SetMuscleTargets(actions);
    }
    
    public void SetPose(MuscleVAECharacter body ,float[] states)
    {
        if (body == _base)
        {
            //_base.SetJointPositions(states.ToList(), targetBones[0].position + new Vector3(0, 0.0f, 0), targetBones[0].rotation);
            _base.ResetPhysics(states.ToList());
        }
        else if (body == _target)
        {
            int count = targetBones.Count;
            for (int i = 0; i < count; i++)
            {
                Vector3 pos = new Vector3(states[i * 13], states[i * 13 + 1], states[i * 13 + 2]);
                Quaternion rot = new Quaternion(states[i * 13 + 3], states[i * 13 + 4], states[i * 13 + 5], states[i * 13 + 6]);
            
                targetBones[i].position = pos;
                targetBones[i].rotation = rot;
                
                bodies[i].targetBone.position = pos;
                bodies[i].targetBone.rotation = rot;
            
            }
            _target.SetPose(states);
        }
    }
    
    public void SetAnimPhase(float phase)
    {
        animator.enabled = true;
        animator.Play(0, -1, phase);
    }
    
    public float GetAnimPhase()
    {
        return animationPhase;
    }

    public bool IsFallen(float threshold)
    {
        if ((baseHead[0].targetBone.position.y - baseHead[0].stateBone.position.y) >= threshold)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    
    public float CallRwd(List<float> baseObs, List<float> targetObs, float dt)
    {
        float error = pose_err(baseObs, targetObs, dt = dt);
        float rwd = Mathf.Exp(-error / 20);
        return rwd;
    }

    
    
    #region utils
    
    [ContextMenu("Fill Maps")]
    public void FillMaps()
    {
        if (root.Count == 0)
        {
            Debug.LogError("No roots found");
            return;
        }
        
        bodies.Clear();
        endEffectors.Clear();
        baseHead.Clear();
        muscles.Clear();
        baseMuscles.Clear();
        targetMuscles.Clear();
        
        var baseBones   = root[0].stateBone.GetComponentsInChildren<Transform>(true);
        var targetBones = root[0].targetBone.GetComponentsInChildren<Transform>(true);
        
        var targetByName = new Dictionary<string, Transform>(StringComparer.OrdinalIgnoreCase);
        foreach (var t in targetBones)
            if (t) targetByName[t.name] = t;

        foreach (var b in baseBones)
        {
            if (!b) continue;

            // exclude?
            bool exclude = false;
            for (int mu = 0; mu < (excludeKeyWords?.Length ?? 0); mu++)
            {
                var kw = excludeKeyWords[mu];
                if (!string.IsNullOrEmpty(kw) && b.name.Contains(kw, StringComparison.OrdinalIgnoreCase))
                {
                    exclude = true;
                    break;
                }
            }
            if (exclude) continue;

            if (!targetByName.TryGetValue(b.name, out var t))
                continue; 
            
            float weight = 1f;
            for (int j = 0; j < (weightKeyWords?.Length ?? 0); j++)
            {
                var kw = weightKeyWords[j];
                if (!string.IsNullOrEmpty(kw) && b.name.Contains(kw, StringComparison.OrdinalIgnoreCase))
                {
                    weight = 0.05f;
                    break;
                }
            }

            bodies.Add(new BoneMapEntry { stateBone = b, targetBone = t, weight = weight });

            bool isEE = false;
            for (int k = 0; k < (eeKeyWords?.Length ?? 0); k++)
            {
                var kw = eeKeyWords[k];
                if (!string.IsNullOrEmpty(kw) &&
                    b.name.Contains(kw, StringComparison.OrdinalIgnoreCase))
                {
                    isEE = true;
                    break;
                }
            }
            bool isHead = false;
            for (int k = 0; k < (headkeyWords?.Length ?? 0); k++)
            {
                var kw = headkeyWords[k];
                if (!string.IsNullOrEmpty(kw) && b.name.Contains(kw, StringComparison.OrdinalIgnoreCase))
                {
                    isHead = true;
                    break;
                }
            }

            if (isHead)
            {
                baseHead.Add(new BoneMapEntry { stateBone = b, targetBone = t, weight = weight });
            }

            bool hasWeightKW = false;
            for (int j = 0; j < (weightKeyWords?.Length ?? 0); j++)
            {
                var kw = weightKeyWords[j];
                if (!string.IsNullOrEmpty(kw) &&
                    b.name.Contains(kw, StringComparison.OrdinalIgnoreCase))
                {
                    hasWeightKW = true;
                    break;
                }
            }

            if (isEE && !hasWeightKW)
            {
                endEffectors.Add(new BoneMapEntry { stateBone = b, targetBone = t, weight = 1f });
            }
        }
        
        var bMuscles = BaseMuscleGroup.GetComponentsInChildren<MuscleTendonUnit>(true);
        var tMuscles = TargetMuscleGroup.GetComponentsInChildren<MuscleTendonUnit>(true);
        Debug.Assert(bMuscles.Length == tMuscles.Length);
        int count = Mathf.Min(bMuscles.Length, tMuscles.Length);
        for (int i = 0; i < count; i++)
        {
            muscles.Add(new MuscleMapEntry{stateMuscle = bMuscles[i], targetMuscle = tMuscles[i], weight = 1.0f});
            baseMuscles.Add(bMuscles[i]);
            targetMuscles.Add(tMuscles[i]);
        }
        //Debug.Log(bMuscles.Length);
        //Debug.Log(tMuscles.Length);
        Debug.Log(baseMuscles.Count);
    }
    #endregion
}
