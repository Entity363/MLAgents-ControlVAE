using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using Unity.MLAgents.Actuators;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;
using static ControlVAE.Utils.MotionUtils;

[DefaultExecutionOrder(-200)]
public class ArticulationControllerCVAE : MonoBehaviour
{
    
    [Serializable]
    public class BoneMapEntry
    {
        public Transform stateBone;
        public Transform targetBone;
        public float weight = 1.0f;
        
    }
    [Header("Characters")]
    public ArticulationCharacterCVAE _base { get; private set; }
    public ArticulationCharacterCVAE _target { get; private set; }
    
    
    [Header("Rigid Bodies")]
    public List<BoneMapEntry> bodies = new();
    
    [Header("EndEffectors")]
    public List<BoneMapEntry> endEffectors = new();
    
    [Header("Roots")]
    public List<BoneMapEntry> root = new();
    
    [Header("Head")]
    public List<BoneMapEntry> baseHead = new();
    
    [Header("Animator")]
    public Animator animator;
    
    
    [Header("Keywords")]
    public string[] weightKeyWords = {"index", "middle", "ring", "pinky", "thumb"};
    public string[] eeKeyWords = {"hand", "foot", "head"};
    public string[] excludeKeyWords = {"collider", "cube", "toe_end", "4", "_end", "alpha"};
    public string[] headkeyWords = {"Head"};


    #region caches
    List<Transform> baseBones = new List<Transform>();
    List<Transform> targetBones = new List<Transform>();

    private float animationPhase = 0f;
    #endregion
    
    
    private void Awake()
    {
        _base = new ArticulationCharacterCVAE();
        _target = new ArticulationCharacterCVAE();
        
        BuildCaches();
        _base.CountActions();
        Debug.Log($"Found {baseBones.Count} base bones, {targetBones.Count} target bones.");
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
        
        foreach (var b in bodies)
        {
            baseBones.Add(b.stateBone);
            targetBones.Add(b.targetBone);
        }
        
        _base.BuildCaches(baseBones, root[0].stateBone, baseHead[0].stateBone);
        _target.BuildCaches(targetBones, root[0].targetBone, baseHead[0].targetBone);
        
        animationPhase = animator.GetCurrentAnimatorStateInfo(0).normalizedTime;
    }

    public void UpdateCaches()
    {
        _base.UpdateCaches(baseBones, root[0].stateBone, baseHead[0].stateBone);
        _target.UpdateCaches(targetBones, root[0].targetBone, baseHead[0].targetBone);
        
        animationPhase = animator.GetCurrentAnimatorStateInfo(0).normalizedTime;
    }

    public void SetTargets(ActionBuffers buffers)
    {
        List<float> actions = new List<float>();
        var acts = buffers.ContinuousActions;
        for (int i = 0; i < acts.Length; i++)
        {
            actions.Add(acts[i]);
        }
        _base.SetTargets(actions);
    }

    public void SetPose(ArticulationCharacterCVAE body ,float[] states)
    {
        if (body == _base)
        {
            //_base.SetJointPositions(states.ToList(), targetBones[0].position + new Vector3(0, 0.0f, 0), targetBones[0].rotation);
            _base.SetBodiesPositions(states.ToList());
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

    public void SetPose_old(ArticulationCharacterCVAE body ,float[] states)
    {
        int count = 0;
        if (body == _base)
        {
            count = baseBones.Count;
            for (int i = 0; i < count; i++)
            {
                Vector3 pos = new Vector3(states[i * 13], states[i * 13 + 1], states[i * 13 + 2]);
                Quaternion rot = new Quaternion(states[i * 13 + 3], states[i * 13 + 4], states[i * 13 + 5], states[i * 13 + 6]);
            
                baseBones[i].position = pos;
                baseBones[i].rotation = rot;
                
                bodies[i].stateBone.position = pos;
                bodies[i].stateBone.rotation = rot;
            
            }
            _base.SetArticulationFromTarget(targetBones.ToArray());
        }
        else if (body == _target)
        {
            count = targetBones.Count;
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
        else
        {
            Debug.Assert(body == _base);
            Debug.Assert(body == _target);
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
    }
    #endregion
}
