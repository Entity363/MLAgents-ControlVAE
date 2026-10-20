using System;
using System.Collections.Generic;
using System.ComponentModel;
using Unity.MLAgents.Actuators;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;
using static ControlVAE.Utils.MotionUtils;

[DefaultExecutionOrder(-200)]
public class ControlVAEController : MonoBehaviour
{
    [Serializable]
    public class BoneMapEntry
    {
        public Transform stateBone;
        public Transform targetBone;
        public float weight = 1.0f;
        
    }
    [Header("Characters")]
    //public Character _base;
    //public Character _target;
    
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

    //Caches
    private readonly List<ConfigurableJoint> _joints = new();
    private readonly List<float> _jointTargets = new();
    private readonly List<float> _jointPrevTargets = new(); //Reading joint targets is a pain and I swear one day I'll use the weyl tensor
    private readonly List<Quaternion> _jointStartRotations = new();
    
    private readonly List<Vector3> _baseBonePositions = new();
    private readonly List<Vector3> _targetBonePositions = new();
    
    private readonly List<Quaternion> _baseBoneRotations = new();
    private readonly List<Quaternion> _targetBoneRotations = new();
    private readonly List<Quaternion> _targetBoneLocalRotations = new();
    
    private readonly List<Vector3> _baseBoneVels = new();
    private readonly List<Vector3> _targetBoneVels = new();
    private readonly List<List<float>> _baseBoneVels6D = new();
    private readonly List<List<float>> _targetBoneVels6D = new();
    
    private readonly List<Rigidbody> _baseRbs = new();
    private readonly List<Rigidbody> _targetRbs = new();
    
    private readonly List<Vector3> _baseBoneAvels = new();
    private readonly List<Vector3> _targetBoneAvels = new();
    
    private readonly List<Vector3> _baseBoneLastPos = new();
    private readonly List<Vector3> _targetBoneLastPos = new();
    private readonly List<Quaternion> _baseBoneLastRot = new();
    private readonly List<Quaternion> _targetBoneLastRot = new();
    
    private readonly List<Vector3> _baseBoneNumericalVels = new();
    private readonly List<Vector3> _targetBoneNumericalVels = new();
    private readonly List<Vector3> _baseBoneNumericalAvels = new();
    private readonly List<Vector3> _targetBoneNumericalAvels = new();
    
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
    private float animationLength = 0;
    private int frame = 0;
    
    private float rwd = 0;
    private List<float> rwds = new();


    private List<float> stateBase = new List<float>();
    private List<float> stateTarget = new List<float>();
    private List<List<float>> obsList = new();
    private List<float> obs = new();
    private List<List<float>> obsListTarget = new();
    private List<float> obsTarget = new();
    
    #endregion
    
    void Awake()
    {
        BuildCaches();
        Debug.Log($"_joints.Count = {_joints.Count}, expected actions = {_joints.Count * 4}");
        //SetAnimPhase(.6f);
        
        stateBase.Clear();
        stateTarget.Clear();
        obsList.Clear();
        obs.Clear();
        obsListTarget.Clear();
        obsTarget.Clear();
        rwds.Clear();
        rwd = 0;

        foreach (var joint in _joints)
        {
            _jointStartRotations.Add(joint.transform.localRotation);
        }
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
        GetStatesBase(stateBase);
        obsList = ComputeObservationsBase();
        ComputeObservations(obs);
        
        GetStatesTarget(stateTarget);
        obsListTarget = ComputeObservationsTarget();
        ComputeTargetObservations(obsTarget);
        
        rwd = CallRwd(obsList,  obsListTarget, Time.fixedDeltaTime);
        rwds.Add(rwd);
        //SetManualTarget(stateBase);
    }
    
    public float GetRwd()
    {
        return rwd;
    }

    private void Update()
    {
        //Debug.Log($"frame: {frame}");
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
    
    //from https://github.com/sergioabreu-g/physical-animations/blob/main/PruebasTFG/Assets/Scripts/BodyPart.cs
    public void SetTargetRotations(List<float> action)
    {
        Debug.Assert(action.Count == _joints.Count * 3);

        int k = 0;
        for (int j = 0; j < _joints.Count; j++)
        {
            float x = action[k];
            float y = action[k + 1];
            float z = action[k + 2];
            
            _jointTargets[k]     = x;
            _jointTargets[k + 1] = y;
            _jointTargets[k + 2] = z;
            
            float tx = (x + 1f) * 0.5f;
            float ty = (y + 1f) * 0.5f;
            float tz = (z + 1f) * 0.5f;
            
            var xRot = Mathf.Lerp(_joints[j].lowAngularXLimit.limit, _joints[j].highAngularXLimit.limit, x);
            var yRot = Mathf.Lerp(-_joints[j].angularYLimit.limit, _joints[j].angularYLimit.limit, y);
            var zRot = Mathf.Lerp(-_joints[j].angularZLimit.limit, _joints[j].angularZLimit.limit, z);

            var rotation = Quaternion.Euler(xRot, yRot, zRot);
            _joints[j].targetRotation = FlipQuatByW(rotation);
            k += 3;
        }
    }
    
    //closer to controlvae
    public void ApplyRotVecActions(List<float> action, float actionScale = 1.0f)
    {
        int k = 0;

        for (int i = 0; i < _joints.Count; i++)
        {
            Vector3 rotVec = new Vector3(action[k], action[k + 1], action[k + 2]) * actionScale;

            Quaternion startRot = _jointStartRotations[i];

            Quaternion q = RotVecToQuat(rotVec);
            q = FlipQuatByW(q);

            // desired rotation relative to rest pose
            Quaternion desired = startRot * q;

            // convert to joint target space
            _joints[i].targetRotation = Quaternion.Inverse(desired);

            k += 3;
        }
    }
    
    //straight from mlagents
    public void SetJointTargetRotation(List<float> actions, float max = 1f)
    {
        int k = 0;
        for (int i = 0; i < _joints.Count; i++)
        {
            var x = actions[k] * max;
            var y =  actions[k + 1] * max;
            var z =  actions[k + 2] * max;
            
            x = (x + 1f) * 0.5f;
            y = (y + 1f) * 0.5f;
            z = (z + 1f) * 0.5f;
            
            var xRot = Mathf.Lerp(_joints[i].lowAngularXLimit.limit, _joints[i].highAngularXLimit.limit, x);
            var yRot = Mathf.Lerp(-_joints[i].angularYLimit.limit, _joints[i].angularYLimit.limit, y);
            var zRot = Mathf.Lerp(-_joints[i].angularZLimit.limit, _joints[i].angularZLimit.limit, z);
            
            _joints[i].targetRotation = Quaternion.Euler(xRot, yRot, zRot);

            k += 3;
        }
    }
    public void SetJointTargetRotationWithMaxForce(List<float> actions, float max = 1f)
    {
        int k = 0;
        for (int i = 0; i < _joints.Count; i++)
        {
            var x = actions[k] * max;
            var y =  actions[k + 1] * max;
            var z =  actions[k + 2] * max;
            
            x = (x + 1f) * 0.5f;
            y = (y + 1f) * 0.5f;
            z = (z + 1f) * 0.5f;
            
            var xRot = Mathf.Lerp(_joints[i].lowAngularXLimit.limit, _joints[i].highAngularXLimit.limit, x);
            var yRot = Mathf.Lerp(-_joints[i].angularYLimit.limit, _joints[i].angularYLimit.limit, y);
            var zRot = Mathf.Lerp(-_joints[i].angularZLimit.limit, _joints[i].angularZLimit.limit, z);
            
            _joints[i].targetRotation = Quaternion.Euler(xRot, yRot, zRot);

            float maxJointForceLimit = 20000f;
            float maxJointSpring = 40000f;
            float maxJointDampen = 5000f;
            float f = actions[k + 3];
            var rawVal = (f + 1f) * 0.5f * maxJointForceLimit;
            var jd = new JointDrive
            {
                positionSpring = maxJointSpring,
                positionDamper = maxJointDampen,
                maximumForce = rawVal
            };
            _joints[i].slerpDrive = jd;

            k += 4;
        }
    }
    
    //Old one
    public void SetJointTargetsController(List<float> actions, float actionScale = 60.3f)
    {
        int k = 0;
        for (int i = 0; i < _joints.Count; i++)
        {
            float x = actions[k] * actionScale;
            float y = actions[k + 1] * actionScale;
            float z = actions[k + 2] * actionScale;
            
            Vector3 rotVec = new Vector3(actions[k], actions[k + 1], actions[k + 2]) * actionScale;
            Quaternion qOffset = RotVecToQuat(rotVec);
            
            Quaternion qTargetLocal = _jointStartRotations[i] * qOffset;
            

            _joints[i].SetTargetRotationLocalDrive(qTargetLocal, _jointStartRotations[i]);

            //_joints[i].SetTargetRotationLocalDrive(qTargetLocal, startLocal);

            k += 3;
        }
    }

    public void SetManualTarget(float[] states)
    {
        //Assert.IsTrue(states.Length == _targetRbs.Count * 13);

        for (int i = 0; i < _targetRbs.Count; i++)
        {
            //_targetRbs[i].Sleep();
            _targetRbs[i].position = new Vector3(states[i * 13], states[i * 13 + 1], states[i * 13 + 2]);
            _targetRbs[i].rotation = new Quaternion(states[i * 13 + 3], states[i * 13 + 4], states[i * 13 + 5], states[i * 13 + 6]);
            //_targetRbs[i].linearVelocity = new Vector3(states[i * 13 + 7], states[i * 13 + 8], states[i * 13 + 9]);
            //_targetRbs[i].angularVelocity = new Vector3(states[i * 13 + 10], states[i * 13 + 11], states[i * 13 + 12]);
            //_targetRbs[i].WakeUp();
            
            _targetBonePositions[i] = _targetRbs[i].position;
            _targetBoneRotations[i] = _targetRbs[i].rotation;
            _targetBoneLocalRotations[i] = _targetRbs[i].transform.localRotation;
            _targetBoneVels[i] = _targetRbs[i].linearVelocity;
            _targetBoneAvels[i] = _targetRbs[i].angularVelocity;
            _targetBoneNumericalVels[i] = GetNumericalVelocity(_targetBonePositions[i], _targetBoneLastPos[i]);
            _targetBoneNumericalAvels[i] = GetNumericalAngularVelocity(_targetBoneRotations[i], _targetBoneLastRot[i]);
            
            _targetBoneLastPos[i] = _targetBonePositions[i];
            _targetBoneLastRot[i] = _targetBoneRotations[i];
        }
    }
    public void SetManualTarget(List<float> states)
    {
        //Assert.IsTrue(states.Count == _targetRbs.Count * 13);

        for (int i = 0; i < _targetRbs.Count; i++)
        {
            //_targetRbs[i].Sleep();
            _targetRbs[i].position = new Vector3(states[i * 13], states[i * 13 + 1], states[i * 13 + 2]);
            _targetRbs[i].rotation = new Quaternion(states[i * 13 + 3], states[i * 13 + 4], states[i * 13 + 5], states[i * 13 + 6]);
            //_targetRbs[i].linearVelocity = new Vector3(states[i * 13 + 7], states[i * 13 + 8], states[i * 13 + 9]);
            //_targetRbs[i].angularVelocity = new Vector3(states[i * 13 + 10], states[i * 13 + 11], states[i * 13 + 12]);
            //_targetRbs[i].WakeUp();
            
            _targetBonePositions[i] = _targetRbs[i].position;
            _targetBoneRotations[i] = _targetRbs[i].rotation;
            _targetBoneLocalRotations[i] = _targetRbs[i].transform.localRotation;
            _targetBoneVels[i] = _targetRbs[i].linearVelocity;
            _targetBoneAvels[i] = _targetRbs[i].angularVelocity;
            _targetBoneNumericalVels[i] = GetNumericalVelocity(_targetBonePositions[i], _targetBoneLastPos[i]);
            _targetBoneNumericalAvels[i] = GetNumericalAngularVelocity(_targetBoneRotations[i], _targetBoneLastRot[i]);
            
            _targetBoneLastPos[i] = _targetBonePositions[i];
            _targetBoneLastRot[i] = _targetBoneRotations[i];
        }
    }

    public void ResetPhysicsCharacter()
    {
        _targetRootRb.transform.parent.position = Vector3.zero;
        _targetRootRb.transform.parent.rotation = Quaternion.identity;
        
        int count = Mathf.Min(_baseRbs.Count, _targetRbs.Count);
        for (int i = 0; i < count; i++)
        {
            //_baseRbs[i].Sleep();
            //_baseRbs[i].isKinematic = true;
        }
        for (int i = 0; i < count; i++)
        {
            _baseRbs[i].position =  _targetBonePositions[i];
            _baseRbs[i].rotation = _targetBoneRotations[i];
            if (i == 0)
            {
                //_baseRbs[0].transform.position = _targetRootRb.position;
                //_baseRbs[0].transform.rotation = Quaternion.identity;
            }
            else
            {
                //_baseRbs[i].transform.localRotation = _jointStartRotations[i-1];
                _jointStartRotations[i - 1] =  _targetBoneLocalRotations[i];
            }
            
            _baseRbs[i].linearVelocity = Vector3.zero;
            _baseRbs[i].angularVelocity = Vector3.zero;
        }
        for (int i = 0; i < count; i++)
        {
            //_baseRbs[i].isKinematic = false;
            //_baseRbs[i].WakeUp();
        }
        
        for (int i = 0; i < _jointTargets.Count; i++)
        {
            _jointPrevTargets[i] = _jointTargets[i] = 0f;
        }

        for (int i = 1; i < _targetRbs.Count; i++)
        {
            //_jointStartRotations[i-1] =  _targetRbs[i].rotation;
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
        animator.enabled = true;
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
        _targetBoneLocalRotations.Clear();
        _baseRbs.Clear();
        _targetRbs.Clear();
        _baseBoneVels.Clear();
        _targetBoneVels.Clear();
        _baseBoneVels6D.Clear();
        _targetBoneVels6D.Clear();
        _baseBoneAvels.Clear();
        _targetBoneAvels.Clear();
        _baseBoneLastPos.Clear();
        _baseBoneLastRot.Clear();
        _baseBoneNumericalVels.Clear();
        _baseBoneNumericalAvels.Clear();
        _targetBoneNumericalVels.Clear();
        _targetBoneNumericalAvels.Clear();
        _targetBoneLastPos.Clear();
        _targetBoneLastRot.Clear();
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
                
                _baseBoneLastPos.Add(_baseRbs[i].position);
                _baseBoneNumericalVels.Add(GetNumericalVelocity(_baseBonePositions[i], _baseBoneLastPos[i], Time.fixedDeltaTime));
                _baseBoneLastRot.Add(_baseRbs[i].rotation);
                _baseBoneNumericalAvels.Add(GetNumericalAngularVelocity(_baseBoneRotations[i], _baseBoneLastRot[i], Time.fixedDeltaTime));
                
                //6d vels
                List<float> a = Compute6DRotation(_baseRbs[i].rotation);
                _baseBoneVels6D.Add(a);
                
                //targets
                _targetRbs.Add(trb);
                _targetBonePositions.Add(_targetRbs[i].position);
                _targetBoneRotations.Add(_targetRbs[i].rotation);
                _targetBoneLocalRotations.Add(_targetRbs[i].transform.localRotation);
                _targetBoneVels.Add(_targetRbs[i].linearVelocity);
                _targetBoneAvels.Add(_targetRbs[i].angularVelocity);
                
                _targetBoneLastPos.Add(_targetRbs[i].position);
                _targetBoneNumericalVels.Add(GetNumericalVelocity(_targetBonePositions[i], _targetBoneLastPos[i], Time.fixedDeltaTime));
                _targetBoneLastRot.Add(_targetRbs[i].rotation);
                _targetBoneNumericalAvels.Add(GetNumericalAngularVelocity(_targetBoneRotations[i], _targetBoneLastRot[i], Time.fixedDeltaTime));
                
                //6d vels
                List<float> b = Compute6DRotation(_targetRbs[i].rotation);
                _targetBoneVels6D.Add(b);
                
                //weights
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
        
        /*
        var st = animator.GetCurrentAnimatorStateInfo(0);

        // phase within current loop
        float phase = st.normalizedTime - Mathf.Floor(st.normalizedTime); // 0..1

        var clipInfo = animator.GetCurrentAnimatorClipInfo(0);
        var c = clipInfo[0].clip;

        int N = Mathf.RoundToInt(c.length * c.frameRate);   // total frames
        frame = Mathf.Clamp(Mathf.FloorToInt(phase * N), 0, N - 1);
        */
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
            _targetBoneLocalRotations[i] = _targetRbs[i].transform.localRotation;
            
            //Debug.Log($"Difference: {_baseRbs[i].rotation * Quaternion.Inverse(_baseRbs[i].transform.rotation)}");
            
            _baseBoneVels[i] = (_baseRbs[i].linearVelocity);
            _targetBoneVels[i] = (_targetRbs[i].linearVelocity);
            
            _baseBoneAvels[i] = (_baseRbs[i].angularVelocity);
            _targetBoneAvels[i] = (_targetRbs[i].angularVelocity);
            
            List<float> a = Compute6DRotation(_baseBoneRotations[i]);
            _baseBoneVels6D[i] = a;

            List<float> b = Compute6DRotation(_targetBoneRotations[i]);
            _targetBoneVels6D[i] = b;
            
            _baseBoneNumericalVels[i] = GetNumericalVelocity(_baseBonePositions[i], _baseBoneLastPos[i], Time.fixedDeltaTime);
            _baseBoneLastPos[i] = _baseBonePositions[i];
            _baseBoneNumericalAvels[i] = GetNumericalAngularVelocity(_baseBoneRotations[i], _baseBoneLastRot[i], Time.fixedDeltaTime);
            _baseBoneLastRot[i] =  _baseBoneRotations[i];
            
            _targetBoneNumericalVels[i] = GetNumericalVelocity(_targetBonePositions[i], _targetBoneLastPos[i], Time.fixedDeltaTime);
            _targetBoneLastPos[i] = _targetBonePositions[i];
            _targetBoneNumericalAvels[i] = GetNumericalAngularVelocity(_targetBoneRotations[i], _targetBoneLastRot[i], Time.fixedDeltaTime);
            _targetBoneLastRot[i] =  _targetBoneRotations[i];
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

    public List<Vector3> GetPositions()
    {
        return _targetBonePositions;
    }

    public List<Quaternion> GetRotations()
    {
        return _targetBoneRotations;
    }

    public List<Vector3> GetVelocities()
    {
        return _targetBoneVels;
    }

    public List<List<float>> GetVelocities6D()
    {
        return _targetBoneVels6D;
    }

    public List<Vector3> GetAngualVelocities()
    {
        return _targetBoneAvels;
    }

    public List<float> GetHeights()
    {
        int count = _targetBonePositions.Count;
        var buffer =  new List<float>();
        for (int i = 0; i < count; i++)
        {
            buffer.Add(_targetBonePositions[i].y);
        }
        return buffer;
    }

    public float GetAnimPhase()
    {
        return animationPhase;
    }

    public List<float> GetUpDirs()
    {
        List<float> buffer = new List<float>();
        
        int count = _targetBoneRotations.Count;
        Quaternion targetHeadingInv = ComputeHeadingInv(_targetRootRotation);
        for (int i = 0; i < count; i++)
        {
            Vector3 bup = (targetHeadingInv * _targetBoneRotations[i]) * Vector3.up;
            float upAlign = Vector3.Dot(bup, Vector3.up);
            buffer.Add(upAlign);
        }
        return buffer;
    }

    #region obs + reward

    public List<List<float>> GetObservationsList()
    {
        return obsList;
    }
    public List<float> GetObservations()
    {
        return obs;
    }

    public List<float> GetStateBase()
    {
        return stateBase;
    }

    public List<float> GetStateTarget()
    {
        return stateTarget;
    }

    public List<List<float>> GetObservationsTargetList()
    {
        return obsListTarget;
    }
    public List<float> GetObservationsTarget()
    {
        return  obsTarget;
    }
    
    public void ComputeObservations(List<float> buffer)
    {
        buffer.Clear();

        
        foreach (var body in obsList)
        {
            foreach (var obs in body)
            {
                buffer.Add(obs);
            }
        }
        //var x = ComputeObservationsTarget();
        //buffer.AddRange(x);
        
        //phase of anim
        //buffer.Add(animationPhase);
    }

    public List<List<float>> ComputeObservationsBase()
    {
        //Debug.LogError("Fix Obs to match state2ob 16 x n + 3");
        Quaternion root_rot_inv = Quaternion.Inverse(_baseRootRotation);
        
        List<List<float>> obs = new List<List<float>>();
        int count = _targetRbs.Count;
        for (int i = 0; i < count; i++)
        {
            List<float> buffer = new List<float>();
            Vector3 pos = _baseBonePositions[i];
            Quaternion rot = _baseBoneRotations[i];
            Vector3 vel = _baseBoneNumericalVels[i]; //_baseBoneVels[i];
            Vector3 avel = _baseBoneNumericalAvels[i]; //_baseBoneAvels[i];

            Vector3 local_pos = BroadcastQuatApply(root_rot_inv, pos - _baseRootPosition);
            Vector3 local_vel =  BroadcastQuatApply(root_rot_inv, vel);
            Vector3 local_avel = BroadcastQuatApply(root_rot_inv, avel);
            Quaternion local_rot = FlipQuatByW(BroadcastQuatMultiply(root_rot_inv, rot));
            var local_rot6D = Compute6DRotation(local_rot);
            float height = pos.y;
            
            //3 pos + 6 rot + 3 vel + 3 avel + 1 height = 16
            buffer.Add(local_pos.x); buffer.Add(local_pos.y); buffer.Add(local_pos.z);
            buffer.Add(local_rot6D[0]); buffer.Add(local_rot6D[1]); buffer.Add(local_rot6D[2]); buffer.Add(local_rot6D[3]); buffer.Add(local_rot6D[4]);  buffer.Add(local_rot6D[5]); 
            buffer.Add(local_vel.x); buffer.Add(local_vel.y); buffer.Add(local_vel.z);
            buffer.Add(local_avel.x); buffer.Add(local_avel.y); buffer.Add(local_avel.z);
            buffer.Add(height);
            
            obs.Add(buffer);
        }
        Vector3 updDir = new Vector3(0, 1, 0);
        Vector3 localUpDir = BroadcastQuatApply(root_rot_inv, updDir);
        List<float> localUpDirs = new List<float>();
        localUpDirs.Add(localUpDir.x); localUpDirs.Add(localUpDir.y); localUpDirs.Add(localUpDir.z);
        
        obs.Add(localUpDirs);
        

        return obs;
    }

    public void GetStatesBase(List<float> buffer)
    {
        buffer.Clear();
        int count = _baseRbs.Count; // should match lists
        for (int i = 0; i < count; i++)
        {
            Vector3 p = _baseBonePositions[i];
            Quaternion q = _baseBoneRotations[i];
            Vector3 v = _baseBoneNumericalVels[i]; //_baseBoneVels
            Vector3 w = _baseBoneNumericalAvels[i]; //_baseBoneAvels
            
            buffer.Add(p.x); buffer.Add(p.y); buffer.Add(p.z);
            buffer.Add(q.x); buffer.Add(q.y); buffer.Add(q.z); buffer.Add(q.w);
            buffer.Add(v.x); buffer.Add(v.y); buffer.Add(v.z);
            buffer.Add(w.x); buffer.Add(w.y); buffer.Add(w.z);
        }
    }

    

    public int GetBaseObservationSize()
    {
        return
            _baseBonePositions.Count * 3 +
            _baseBoneRotations.Count * 6 +
            _baseBoneNumericalVels.Count * 3 + // _baseBoneVels
            _baseBoneNumericalAvels.Count * 3 + //_baseBoneAvels
            _baseBonePositions.Count * 1 + //height 
            3; //updir
    }
    
    public int GetBaseStatesSize()
    {
        return
            _baseBonePositions.Count * 3 +
            _baseBoneRotations.Count * 4 +
            _baseBoneNumericalVels.Count * 3 + // _baseBoneVels
            _baseBoneNumericalAvels.Count * 3; //_baseBoneAvels
    }
    
    public void ComputeTargetObservations(List<float> buffer)
    {
        buffer.Clear();

        
        foreach (var body in obsListTarget)
        {
            foreach (var obs in body)
            {
                buffer.Add(obs);
            }
        }
    }
    
    public List<List<float>> ComputeObservationsTarget()
    {
        //Debug.LogError("Fix Obs to match state2ob 16 x n + 3");
        Quaternion root_rot_inv = Quaternion.Inverse(_targetRootRotation);
        
        List<List<float>> obs = new List<List<float>>();
        int count = _targetRbs.Count;
        for (int i = 0; i < count; i++)
        {
            List<float> buffer = new List<float>();
            Vector3 pos = _targetBonePositions[i];
            Quaternion rot = _targetBoneRotations[i];
            Vector3 vel = _targetBoneNumericalVels[i]; //_targetBoneVels
            Vector3 avel = _targetBoneNumericalAvels[i]; //_targetBoneAvels

            Vector3 local_pos = BroadcastQuatApply(root_rot_inv, pos - _targetRootPosition);
            Vector3 local_vel =  BroadcastQuatApply(root_rot_inv, vel);
            Vector3 local_avel = BroadcastQuatApply(root_rot_inv, avel);
            Quaternion local_rot = FlipQuatByW(BroadcastQuatMultiply(root_rot_inv, rot));
            var local_rot6D = Compute6DRotation(local_rot);
            float height = pos.y;
            
            //3 pos + 6 rot + 3 vel + 3 avel + 1 height = 16
            buffer.Add(local_pos.x); buffer.Add(local_pos.y); buffer.Add(local_pos.z);
            buffer.Add(local_rot6D[0]); buffer.Add(local_rot6D[1]); buffer.Add(local_rot6D[2]); buffer.Add(local_rot6D[3]); buffer.Add(local_rot6D[4]);  buffer.Add(local_rot6D[5]); 
            buffer.Add(local_vel.x); buffer.Add(local_vel.y); buffer.Add(local_vel.z);
            buffer.Add(local_avel.x); buffer.Add(local_avel.y); buffer.Add(local_avel.z);
            buffer.Add(height);
            
            obs.Add(buffer);
        }
        Vector3 updDir = new Vector3(0, 1, 0);
        Vector3 localUpDir = BroadcastQuatApply(root_rot_inv, updDir);
        List<float> localUpDirs = new List<float>();
        localUpDirs.Add(localUpDir.x); localUpDirs.Add(localUpDir.y); localUpDirs.Add(localUpDir.z);
        
        obs.Add(localUpDirs);
        

        return obs;
    }
    
    public void GetStatesTarget(List<float> buffer)
    {
        buffer.Clear();
        int count = _targetRbs.Count; // should match lists
        for (int i = 0; i < count; i++)
        {
            Vector3 p = _targetBonePositions[i];
            Quaternion q = _targetBoneRotations[i];
            Vector3 v = _targetBoneNumericalVels[i]; //_targetBoneVels
            Vector3 w = _targetBoneNumericalAvels[i]; //_targetBoneAvels
            
            buffer.Add(p.x); buffer.Add(p.y); buffer.Add(p.z);
            buffer.Add(q.x); buffer.Add(q.y); buffer.Add(q.z); buffer.Add(q.w);
            buffer.Add(v.x); buffer.Add(v.y); buffer.Add(v.z);
            buffer.Add(w.x); buffer.Add(w.y); buffer.Add(w.z);
        }
    }
    
    public int GetTarComputeObservationsize()
    {
        return
            _targetBonePositions.Count * 3 +
            _targetBoneRotations.Count * 6 +
            _targetBoneNumericalVels.Count * 3 + //_targetBoneVels
            _targetBoneNumericalAvels.Count * 3 + //_targetBoneAvels
            _targetBonePositions.Count * 1 + //height
            3 ; //updir
    }

    public int GetTargetStatesSize()
    {
        return
            _targetBonePositions.Count * 3 +
            _targetBoneRotations.Count * 4 +
            _targetBoneNumericalVels.Count * 3 + //_targetBoneVels
            _targetBoneNumericalAvels.Count * 3; //_targetBoneAvels
    }


    
    public float CallRwd(List<List<float>> baseObs, List<List<float>> targetObs, float dt)
    {
        float error = PoseError(baseObs, targetObs, dt = dt);
        float rwd = Mathf.Exp(-error / 20);
        return rwd;
    }


    public float PoseError(List<List<float>> stateObs, List<List<float>> targetObs, float posW = 0.2f, float rotW = 0.1f, float velW = 0.5f, float avelW = 0.5f, float heightW = 1.2f, float upDirW = 3f, float dt = 1f/20f)
    {
        int Count = Mathf.Min(stateObs.Count, targetObs.Count);
        float posLoss = 0;
        float rotLoss = 0;
        float velLoss = 0;
        float avelLoss = 0;
        float heightLoss = 0;
        float upDirLoss = 0;

        for (int j = 0; j < Count - 1; j++)
        {
            for (int i = 0; i < 3; i++) //pos
            {
                float delta_pos = stateObs[j][i] - targetObs[j][i];
                float p = Mathf.Abs(delta_pos);
                posLoss += p;
            }
            for (int i = 3; i < 9; i++) //rot
            {
                float delta_rot = stateObs[j][i] - targetObs[j][i];
                float r = Mathf.Abs(delta_rot);
                rotLoss += r;
            }
            for (int i = 9; i < 12; i++) //vel
            {
                float delta_vel = stateObs[j][i] - targetObs[j][i];
                float v = Mathf.Abs(delta_vel);
                velLoss += v;
            }
            for (int i = 12; i < 15; i++) //avel
            {
                float delta_avel = stateObs[j][i] - targetObs[j][i];
                float a = Mathf.Abs(delta_avel);
                avelLoss += a;
            }
            //height
            float delta_h =  stateObs[j][15] - targetObs[j][15];
            float h = Mathf.Abs(delta_h);
            heightLoss += h;
        }

        for (int i = 0; i < 3; i++)
        {
            float delta_updir = stateObs[Count - 1][i] -  targetObs[Count - 1][i];
            float u = Mathf.Abs(delta_updir);
            upDirLoss += u;
        }
        
        posLoss /= 3f * Count;
        rotLoss /= 6f * Count;
        velLoss /= 3f * Count;
        avelLoss /= 3f * Count;
        heightLoss /= 1f * Count;
        upDirLoss /= 3f;
        
        float rwd = posW * posLoss + rotW *  rotLoss + velW * velLoss * dt + avelW * avelLoss * dt + heightW * heightLoss + upDirW * upDirLoss;
        return rwd;
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
