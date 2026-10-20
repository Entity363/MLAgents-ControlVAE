using System;
using System.Collections.Generic;
using System.ComponentModel;
using Unity.MLAgents.Actuators;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;

public class DeepMimicController : MonoBehaviour
{
    [Serializable]
    public class BoneMapEntry
    {
        public Transform stateBone;
        public Transform targetBone;
        public float weight = 1.0f;
        
    }
    
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
    public string[] weightKeyWords;
    public string[] eeKeyWords;
    public string[] excludeKeyWords;
    public string[] headkeyWords = {"Head"};

    #region caches

    //Caches
    private readonly List<ConfigurableJoint> _joints = new();
    private readonly List<float> _jointTargets = new();
    private readonly List<float> _jointPrevTargets = new(); //Reading joint targets is a pain and I swear one day I'll use the weyl tensor
    
    private readonly List<Vector3> _baseBonePositions = new();
    private readonly List<Vector3> _targetBonePositions = new();
    
    private readonly List<Quaternion> _baseBoneRotations = new();
    private readonly List<Quaternion> _targetBoneRotations = new();
    
    private readonly List<Vector3> _baseBoneVels = new();
    private readonly List<Vector3> _targetBoneVels = new();
    
    private readonly List<Rigidbody> _baseRbs = new();
    private readonly List<Rigidbody> _targetRbs = new();
    
    private readonly List<Vector3> _baseBoneAvels = new();
    private readonly List<Vector3> _targetBoneAvels = new();
    
    private readonly List<Rigidbody> _baseEndEffectors = new();
    private readonly List<Rigidbody> _targetEndEffectors = new();
    
    private readonly List<Vector3> _baseEndEffectorPositions = new();
    private readonly List<Vector3> _targetEndEffectorPositions = new();

    private Rigidbody _baseRootRb;
    private Rigidbody _targetRootRb;
    
    private Vector3 _baseRootPosition;
    private Vector3 _targetRootPosition;

    private Vector3 _baseRootVelocity;
    private Vector3 _targetRootVelocity;
    
    private Quaternion  _baseRootRotation;
    private Quaternion _targetRootRotation;
    
    private Vector3 _baseHeadPosition;
    private Vector3 _targetHeadPosition;
    
    private readonly List<float> _weights = new();
    private readonly List<float> _endEffectorWeights = new();

    private float animationPhase = 0;
    
    #endregion
    
    void Awake()
    {
        BuildCaches();
        Debug.Log($"_joints.Count = {_joints.Count}, expected actions = {_joints.Count * 3}");
        //SetAnimPhase(.6f);
    }

    void FixedUpdate()
    {
        UpdateCaches();

        /*
        List<float> action = new List<float>(_joints.Count * 3);
        for (int i = 0; i < _joints.Count * 3; i++)
            action.Add(UnityEngine.Random.Range(-1f, 1f));

        SetJointTargets(action);
        */
    }
    
    //Remember to call in OnActionReceived
    public void SetJointTargets(List<float> action)
    {
        Debug.Assert(action.Count == _joints.Count * 3);

        for (int i = 0; i < _jointTargets.Count; i++)
        {
            _jointPrevTargets[i] = _jointTargets[i];
        }

        int k = 0;
        for (int j = 0; j < _joints.Count; j++)
        {
            float x = action[k];
            float y = action[k + 1];
            float z = action[k + 2];

            _jointTargets[k]     = x;
            _jointTargets[k + 1] = y;
            _jointTargets[k + 2] = z;

            Vector3 v = new Vector3(x, y, z);
            float angle = v.magnitude;

            if (angle > 1e-6f)
            {
                Vector3 axis = v / angle;
                _joints[j].targetRotation = Quaternion.AngleAxis(angle * Mathf.Rad2Deg, axis);
            }
            else
            {
                _joints[j].targetRotation = Quaternion.identity;
            };
            k += 3;
        }
    }

    public void ResetPhysicsCharacter()
    {
        _targetRootRb.transform.parent.position = Vector3.zero;
        _targetRootRb.transform.parent.rotation = Quaternion.identity;
        
        int count = Mathf.Min(_baseRbs.Count, _targetRbs.Count);
        for (int i = 0; i < count; i++)
        {
            _baseRbs[i].Sleep();
            _baseRbs[i].isKinematic = true;
        }
        for (int i = 0; i < count; i++)
        {
            _baseRbs[i].position =  _targetRbs[i].transform.position;
            _baseRbs[i].rotation = _targetRbs[i].transform.rotation;
            //_baseRbs[i].linearVelocity = _targetRbs[i].linearVelocity;
            //_baseRbs[i].angularVelocity = _targetRbs[i].angularVelocity;
        }
        for (int i = 0; i < count; i++)
        {
            _baseRbs[i].isKinematic = false;
            _baseRbs[i].WakeUp();
        }
        
        for (int i = 0; i < _jointTargets.Count; i++)
        {
            _jointPrevTargets[i] = _jointTargets[i] = 0f;
        }
        
        UpdateCaches();
    }

    public bool IsFallen(float height)
    {
        if ((_targetHeadPosition - _baseHeadPosition).y > height)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    public void SetAnimPhase(float phase)
    {
        animator.Play(0, -1, phase);
    }
    
    public void BuildCaches()
    {
        _joints.Clear();
        _jointTargets.Clear();
        _jointPrevTargets.Clear();
        _baseBonePositions.Clear();
        _targetBonePositions.Clear();
        _baseBoneRotations.Clear();
        _targetBoneRotations.Clear();
        _baseRbs.Clear();
        _targetRbs.Clear();
        _baseBoneVels.Clear();
        _targetBoneVels.Clear();
        _baseBoneAvels.Clear();
        _targetBoneAvels.Clear();
        _baseEndEffectors.Clear();
        _targetEndEffectors.Clear();
        _baseEndEffectorPositions.Clear();
        _targetEndEffectorPositions.Clear();
        _baseRootPosition = Vector3.zero;
        _targetRootPosition = Vector3.zero;
        _baseRootRotation = Quaternion.identity;
        _targetRootRotation = Quaternion.identity;
        _weights.Clear();
        _endEffectorWeights.Clear();
        _baseHeadPosition = Vector3.zero;
        _targetHeadPosition = Vector3.zero;
        animationPhase = 0;
        
        //pos + rot + vel + avel + ee + com
        int i = 0;
        foreach (var boneMap in bodies)
        {
            var brb = boneMap.stateBone.GetComponent<Rigidbody>();
            var trb = boneMap.targetBone.GetComponent<Rigidbody>();
            if (!brb) Debug.Log($"Missing Rigidbody on {boneMap.stateBone.name}");
            if (!trb) Debug.Log($"Missing Rigidbody on {boneMap.targetBone.name}");
            if (trb && brb)
            {
                _baseRbs.Add(brb);
                _baseBonePositions.Add(_baseRbs[i].position);
                _baseBoneRotations.Add(_baseRbs[i].rotation);
                _baseBoneVels.Add(_baseRbs[i].linearVelocity);
                _baseBoneAvels.Add(_baseRbs[i].angularVelocity);
                
                _targetRbs.Add(trb);
                _targetBonePositions.Add(_targetRbs[i].position);
                _targetBoneRotations.Add(_targetRbs[i].rotation);
                _targetBoneVels.Add(_targetRbs[i].linearVelocity);
                _targetBoneAvels.Add(_targetRbs[i].angularVelocity);

                _weights.Add(boneMap.weight);
                
                var j = boneMap.stateBone.GetComponent<ConfigurableJoint>();
                if (!j)
                {
                    Debug.Log($"Missing ConfigurableJoint on {boneMap.stateBone.name}");
                }
                else
                {
                    _joints.Add(j);
                }
                
                i++;
            }
        }

        for (int j = 0; j < _joints.Count; j++)
        {
            _jointPrevTargets.Add(0f); // x
            _jointPrevTargets.Add(0f); // y
            _jointPrevTargets.Add(0f); // z

            _jointTargets.Add(0f);     // x
            _jointTargets.Add(0f);     // y
            _jointTargets.Add(0f);     // z
        }
        Debug.Assert(
            _jointTargets.Count == _joints.Count * 3,
            $"Joint target size mismatch: {_jointTargets.Count} vs {_joints.Count * 3}"
        );
        

        foreach (var boneMap in endEffectors)
        {
            _baseEndEffectors.Add(boneMap.stateBone.GetComponent<Rigidbody>());
            _targetEndEffectors.Add(boneMap.targetBone.GetComponent<Rigidbody>());
            
            _baseEndEffectorPositions.Add(boneMap.stateBone.position);
            _targetEndEffectorPositions.Add(boneMap.targetBone.position);
            _endEffectorWeights.Add(boneMap.weight);
        }


        foreach (var boneMap in root)
        {
            _baseRootPosition =  boneMap.stateBone.position;
            _targetRootPosition =  boneMap.targetBone.position;
            
            _baseRootRotation   = boneMap.stateBone.rotation;
            _targetRootRotation = boneMap.targetBone.rotation;
            
            _baseRootRb = boneMap.stateBone.GetComponent<Rigidbody>();
            _targetRootRb = boneMap.targetBone.GetComponent<Rigidbody>();
            
            _baseRootVelocity = _baseRootRb.linearVelocity;
            _targetRootVelocity = _targetRootRb.linearVelocity;
        }

        foreach (var boneMap in baseHead)
        {
            _baseHeadPosition =  boneMap.stateBone.position;
            _targetHeadPosition =  boneMap.targetBone.position;
        }
        
        animationPhase = animator.GetCurrentAnimatorStateInfo(0).normalizedTime;
        
    }
    public void UpdateCaches()
    {
        //pos + rot + vel + avel + ee + com
        int count = Mathf.Min(_baseRbs.Count, _targetRbs.Count);
        for (int i = 0; i < count; i++)
        {
            _baseBonePositions[i] = _baseRbs[i].position;
            _baseBoneRotations[i] = _baseRbs[i].rotation;
            
            _targetBonePositions[i] = _targetRbs[i].position;
            _targetBoneRotations[i] = _targetRbs[i].rotation;
            
            _baseBoneVels[i] = (_baseRbs[i].linearVelocity);
            _targetBoneVels[i] = (_targetRbs[i].linearVelocity);
            
            _baseBoneAvels[i] = (_baseRbs[i].angularVelocity);
            _targetBoneAvels[i] = (_targetRbs[i].angularVelocity);
        }
        
        int eeCount = Mathf.Min(_baseEndEffectors.Count, _targetEndEffectors.Count);
        for (int ee = 0; ee < eeCount; ee++)
        { 
            _baseEndEffectorPositions[ee] = (_baseEndEffectors[ee].position);
            _targetEndEffectorPositions[ee] = (_targetEndEffectors[ee].position);
        }

        foreach (var boneMap in root)
        {
            _baseRootPosition =  boneMap.stateBone.position;
            _targetRootPosition =  boneMap.targetBone.position;
            
            _baseRootRotation   = boneMap.stateBone.rotation;
            _targetRootRotation = boneMap.targetBone.rotation;
            
            _baseRootVelocity = _baseRootRb.linearVelocity;
            _targetRootVelocity = _targetRootRb.linearVelocity;
        }
        foreach (var boneMap in baseHead)
        {
            _baseHeadPosition =  boneMap.stateBone.position;
            _targetHeadPosition =  boneMap.targetBone.position;
        }
        
        animationPhase = animator.GetCurrentAnimatorStateInfo(0).normalizedTime;
    }

    #region obs + reward
    public void GetObservations(List<float> buffer)
    {
        buffer.Clear();
        
        GetObservationsBase(buffer);
        GetObservationsTarget(buffer);
        
        //phase of anim
        buffer.Add(animationPhase);
        
    }

    public void GetObservationsBase(List<float> buffer)
    {
        Quaternion baseHeadingInv = ComputeHeadingInv(_baseRootRotation);
        
        //Original DeepMimic uses base positions relative to com + angles + vels + avels
        foreach (var q in _baseBonePositions)
        {
            Vector3 b = baseHeadingInv * (q - _baseRootPosition);
            buffer.Add(b.x);
            buffer.Add(b.y);
            buffer.Add(b.z);
        }
        foreach (var b in _baseBoneRotations)
        {
            var q = baseHeadingInv * b;
            buffer.Add(q.x);
            buffer.Add(q.y);
            buffer.Add(q.z);
            buffer.Add(q.w);
        }
        foreach (var b in _baseBoneVels)
        {
            var q = baseHeadingInv * b;
            buffer.Add(q.x);
            buffer.Add(q.y);
            buffer.Add(q.z);
        }
        foreach (var b in _baseBoneAvels)
        {
            var q = baseHeadingInv * b;
            buffer.Add(q.x);
            buffer.Add(q.y);
            buffer.Add(q.z);
        }
        //hi
        foreach (var b in _baseBonePositions)
        {
            buffer.Add(b.y);
        }
        //root y il llf
        Vector3 bup = (baseHeadingInv * _baseRootRotation) * Vector3.up;
        buffer.Add(bup.y);
    }

    public void GetObservationsTarget(List<float> buffer)
    {
        //changed for consistency
        Quaternion targetHeadingInv = ComputeHeadingInv(_baseRootRotation); //_targetRootRotation 
        //Target obs
        foreach (var q in _targetBonePositions)
        {
            
            Vector3 b = targetHeadingInv * (q - _targetRootPosition);
            buffer.Add(b.x);
            buffer.Add(b.y);
            buffer.Add(b.z);
        }
        foreach (var b in _targetBoneRotations)
        {
            var q = targetHeadingInv * b;
            buffer.Add(q.x);
            buffer.Add(q.y);
            buffer.Add(q.z);
            buffer.Add(q.w);
        }
        foreach (var b in _targetBoneVels)
        {
            var q = targetHeadingInv * b;
            buffer.Add(q.x);
            buffer.Add(q.y);
            buffer.Add(q.z);
        }
        foreach (var b in _targetBoneAvels)
        {
            var q = targetHeadingInv * b;
            buffer.Add(q.x);
            buffer.Add(q.y);
            buffer.Add(q.z);
        }
        //hi
        foreach (var b in _targetBonePositions)
        {
            buffer.Add((b - _targetRootPosition).y);
        }
        //root y il llf
        var tup = (targetHeadingInv * _targetRootRotation) * Vector3.up;
        buffer.Add(tup.y);
        
    }
    public float GetPosDiffSqr(Vector3 basePos, Quaternion baseRot, List<Vector3> basePositions, List<Vector3> targetPositions, List<float> weights)
    {
        var headingInv = ComputeHeadingInv(baseRot);
        float diff = 0.0f;
        float weightSum = 0.0f;

        int count = Mathf.Min(basePositions.Count, targetPositions.Count, weights.Count);
        for (int i = 0; i < count; i++)
        {
            Vector3 baseP   = headingInv * (basePositions[i]   - basePos);
            Vector3 targetP = headingInv * (targetPositions[i] - basePos); //_targetRootRotation

            float w = weights[i];
            diff += (baseP - targetP).sqrMagnitude * w;
            weightSum += w;
        }
        return diff/weightSum;
    }
    
    public float GetVecListDiffSqr(Vector3 baseVec, Quaternion baseRot, List<Vector3> baseVectors, List<Vector3> targetVectors, List<float> weights)
    {
        var headingInv = ComputeHeadingInv(baseRot);
        float diff = 0f;
        float weightSum = 0.0f;
        
        int count = Mathf.Min(baseVectors.Count, targetVectors.Count, weights.Count);
        for (int i = 0; i < count; i++)
        {
            Vector3 b = headingInv * baseVectors[i];
            Vector3 t = headingInv * targetVectors[i];

            float w = weights[i];
            diff += (b - t).sqrMagnitude * w;
            weightSum += w;
        }
        return diff/weightSum;
    }
    
    public float GetQuatDiffSqr(Vector3 basePos, Quaternion baseRot, List<Quaternion> baseRotation, List<Quaternion> targetRotation, List<float> weights)
    {
        var headingInv = ComputeHeadingInv(baseRot);
        float diff = 0.0f;
        float weightSum = 0.0f;
        
        int count = Mathf.Min(baseRotation.Count, targetRotation.Count, weights.Count);
        for (int i = 0; i < count; i++)
        {
            Quaternion headingRot = headingInv * baseRotation[i];
            Quaternion targetRot = headingInv * targetRotation[i];

            Quaternion rotDiff = (headingRot * Quaternion.Inverse(targetRot));
            if (rotDiff.w < 0f)
            {
                rotDiff = new Quaternion(-rotDiff.x, -rotDiff.y, -rotDiff.z, -rotDiff.w);
            }
            
            float ang = Mathf.Clamp(rotDiff.w, -1.0f, 1.0f);
            float angels = 2f * Mathf.Acos(ang);

            float w = weights[i];
            diff += angels*angels * w;
            weightSum += w;
        }
        return diff/weightSum;
    }

    public float GetPositionDiffSqr()
    {
        float diff = GetPosDiffSqr(_baseRootPosition, _baseRootRotation, _baseBonePositions, _targetBonePositions, _weights);
        return diff;
    }

    public float GetRotDiffSqr()
    {
        float diff = GetQuatDiffSqr(_baseRootPosition, _baseRootRotation, _baseBoneRotations, _targetBoneRotations, _weights);
        return diff;
    }

    public float GetVelDiffSqr()
    {
        float diff = GetVecListDiffSqr(_baseRootPosition, _baseRootRotation, _baseBoneVels, _targetBoneVels, _weights);
        return diff;
    }

    public float GetAvelDiffSqr()
    {
        float diff = GetVecListDiffSqr(_baseRootPosition, _baseRootRotation, _baseBoneAvels, _targetBoneAvels, _weights);
        return diff;
    }

    public float GetEeDiffSqr()
    {
        float diff = GetPosDiffSqr(_baseRootPosition, _baseRootRotation,  _baseEndEffectorPositions,  _targetEndEffectorPositions, _endEffectorWeights);
        return diff;
    }

    public float GetComDiffSqr()
    {
        float diff = (_targetRootPosition-_baseRootPosition).sqrMagnitude;
        return diff;
    }

    public float GetHeadingRwd(float desiredSpeed)
    {
        Vector3 heading = (_targetRootPosition-_baseRootPosition);
        heading.y = 0f;
        
        float dist =  heading.sqrMagnitude;
        if (dist < 1e-4f) return 1f;
        heading.Normalize();
        float vAlong = Vector3.Dot(heading, _baseRootVelocity);
        
        float rwd = Mathf.Max(0, desiredSpeed -  vAlong);
        rwd *= rwd;
        return rwd;
    }
    #endregion
    

    #region utils
    static Quaternion ComputeHeadingInv(Quaternion rootRot)
    {
        Vector3 fwd = rootRot * Vector3.forward;
        fwd.y = 0f;
        if (fwd.sqrMagnitude < 1e-8f) fwd = Vector3.forward;
        fwd.Normalize();

        Quaternion heading = Quaternion.LookRotation(fwd, Vector3.up);
        return Quaternion.Inverse(heading);
    }
    
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
