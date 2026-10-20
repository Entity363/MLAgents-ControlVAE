using System;
using System.Collections.Generic;
using System.ComponentModel;
using Unity.MLAgents.Actuators;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;
using static ControlVAE.Utils.MotionUtils;


public class Character : MonoBehaviour
{
    private readonly List<ConfigurableJoint> joints = new();
    private List<float> jointTargets = new();
    private List<float> jointLastTargets = new();
    private List<Rigidbody> rbs = new();
    private List<Vector3> positions = new();
    private List<Quaternion> rotations = new();
    private List<Vector3> velocities = new();
    private List<Vector3> angularVelocities = new();
    private List<List<float>> rotations6D = new();
    private List<Vector3> previousPositions = new();
    private List<Quaternion> previousRotations = new();
    private List<Vector3> numericalVelocities = new();
    private List<Vector3> numericalAngularVelocities = new();
    private List<Rigidbody> head = new();
    private List<Vector3> headPositions = new();
    private List<Rigidbody> endEffectors = new();
    private List<Vector3> endPositions = new();
    private List<float> weights = new();

    private List<Rigidbody> root = new();
    private Vector3 rootPosition = new Vector3(0, 0, 0);
    private Vector3 lastRootPosition = new Vector3(0, 0, 0);
    private Quaternion rootRotation = new Quaternion(0, 0, 0, 1);
    private Quaternion lastRootRotation = new Quaternion(0, 0, 0, 1);
    private Vector3 rootVelocity = new Vector3(0, 0, 0);
    private Vector3 rootAngularVelocity = new Vector3(0, 0, 0);
    private Vector3 rootNumericalVelocity = new Vector3(0, 0, 0);
    private Vector3 rootNumericalAngularVelocity = new Vector3(0, 0, 0);

    public void SetBodies(List<Rigidbody> bodies)
    {
        rbs = bodies;
    }

    public void SetEndEffectors(List<Rigidbody> ee)
    {
        endEffectors = ee;
    }

    public void SetRoot(List<Rigidbody> r)
    {
        root = r;
    }

    public void SetHead(List<Rigidbody> h)
    {
        head = h;
    }

    private void BuildCaches(List<float> w)
    {
        joints.Clear();
        jointLastTargets.Clear();
        jointTargets.Clear();
        rbs.Clear();
        positions.Clear();
        rotations.Clear();
        velocities.Clear();
        angularVelocities.Clear();
        rotations6D.Clear();
        previousPositions.Clear();
        previousRotations.Clear();
        numericalVelocities.Clear();
        numericalAngularVelocities.Clear();
        head.Clear();
        headPositions.Clear();
        endEffectors.Clear();
        endPositions.Clear();
        weights.Clear();
        root.Clear();
        weights.AddRange(w);
        
        foreach (var b in rbs)
        {
            positions.Add(b.position);
            rotations.Add(b.rotation);
            velocities.Add(b.linearVelocity);
            angularVelocities.Add(b.angularVelocity);
            previousPositions.Add(b.position);
            previousRotations.Add(b.rotation);
            List<float> a = Compute6DRotation(b.rotation);
            rotations6D.Add(a);
            numericalVelocities.Add(GetNumericalVelocity(b.position, b.position));
            numericalAngularVelocities.Add(GetNumericalAngularVelocity(b.rotation, b.rotation));
        }

        for (int i = 0; i < joints.Count; i++)
        {
            jointLastTargets.Add(0f); // x
            jointLastTargets.Add(0f); // y
            jointLastTargets.Add(0f); // z

            jointTargets.Add(0f);     // x
            jointTargets.Add(0f);     // y
            jointTargets.Add(0f);     // z
        }

        foreach (var b in endEffectors)
        {
            endPositions.Add(b.position);
        }

        foreach (var b in head)
        {
            headPositions.Add(b.position);
        }

        foreach (var b in root)
        {
            rootPosition = b.position;
            rootRotation = b.rotation;
            rootVelocity = b.linearVelocity;
            rootAngularVelocity = b.angularVelocity;
            rootNumericalVelocity = GetNumericalVelocity(b.position, b.position);
            lastRootPosition = rootPosition;
            rootNumericalAngularVelocity = GetNumericalAngularVelocity(b.rotation, b.rotation);
            lastRootRotation = rootRotation;
        }
    }

    private void UpdateCaches()
    {
        for (int i = 0; i < joints.Count; i++)
        {
            jointLastTargets[i] = jointLastTargets[i];
        }

        for (int i = 0; i < rbs.Count; i++)
        {
            positions[i] = rbs[i].position;
            rotations[i] = rbs[i].rotation;
            List<float> a = Compute6DRotation(rbs[i].rotation);
            rotations6D[i] = a;
            velocities[i] = rbs[i].linearVelocity;
            angularVelocities[i] = rbs[i].angularVelocity;
            numericalVelocities[i] = (GetNumericalVelocity(positions[i], previousPositions[i]));
            numericalAngularVelocities[i] =  (GetNumericalAngularVelocity(rotations[i], previousRotations[i]));
            previousPositions[i] = positions[i];
            previousRotations[i] = rotations[i];
        }

        for (int i = 0; i < endEffectors.Count; i++)
        {
            endPositions[i] = endEffectors[i].position;
        }

        for (int i = 0; i < head.Count; i++)
        {
            headPositions[i] = head[i].position;
        }

        foreach (var b in root)
        {
            rootPosition = b.position;
            rootRotation = b.rotation;
            rootVelocity = b.linearVelocity;
            rootAngularVelocity = b.angularVelocity;
            rootNumericalVelocity = GetNumericalVelocity(b.position, b.position);
            lastRootPosition = rootPosition;
            rootNumericalAngularVelocity = GetNumericalAngularVelocity(b.rotation, b.rotation);
            lastRootRotation = rootRotation;
        }
    }
    
    public void SetManualStates(List<float> states)
    {
        //Assert.IsTrue(states.Count == rbs.Count * 13);

        for (int i = 0; i < rbs.Count; i++)
        {
            //_targetRbs[i].Sleep();
            rbs[i].position = new Vector3(states[i * 13], states[i * 13 + 1], states[i * 13 + 2]);
            rbs[i].rotation = new Quaternion(states[i * 13 + 3], states[i * 13 + 4], states[i * 13 + 5], states[i * 13 + 6]);
            //_targetRbs[i].linearVelocity = new Vector3(states[i * 13 + 7], states[i * 13 + 8], states[i * 13 + 9]);
            //_targetRbs[i].angularVelocity = new Vector3(states[i * 13 + 10], states[i * 13 + 11], states[i * 13 + 12]);
            //_targetRbs[i].WakeUp();
            
            positions[i] = rbs[i].position;
            rotations[i] = rbs[i].rotation;
            velocities[i] = rbs[i].linearVelocity;
            angularVelocities[i] = rbs[i].angularVelocity;
            numericalVelocities[i] = GetNumericalVelocity(positions[i], previousPositions[i]);
            angularVelocities[i] = GetNumericalAngularVelocity(rotations[i], previousRotations[i]);
            
            previousPositions[i] = positions[i];
            previousRotations[i] = rotations[i];
        }
    }

    public List<float> GetStates()
    {
        List<float> buffer = new();
        for (int i = 0; i < rbs.Count; i++)
        {
            Vector3 p = positions[i];
            Quaternion q = rotations[i];
            Vector3 v = velocities[i];
            Vector3 w = angularVelocities[i];
            
            buffer.Add(p.x); buffer.Add(p.y); buffer.Add(p.z);
            buffer.Add(q.x); buffer.Add(q.y); buffer.Add(q.z); buffer.Add(q.w);
            buffer.Add(v.x); buffer.Add(v.y); buffer.Add(v.z);
            buffer.Add(w.x); buffer.Add(w.y); buffer.Add(w.z);
        }
        return buffer;
    }
    
    public List<List<float>> ComputeObservations()
    {
        //Debug.LogError("Fix Obs to match state2ob 16 x n + 3");
        Quaternion root_rot_inv = Quaternion.Inverse(rootRotation);
        
        List<List<float>> obs = new List<List<float>>();
        int count = rbs.Count;
        for (int i = 0; i < count; i++)
        {
            List<float> buffer = new List<float>();
            Vector3 pos = positions[i];
            Quaternion rot = rotations[i];
            Vector3 vel = numericalVelocities[i]; //_targetBoneVels
            Vector3 avel = numericalAngularVelocities[i]; //_targetBoneAvels

            Vector3 local_pos = BroadcastQuatApply(root_rot_inv, pos - rootPosition);
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

    public List<float> ComputeObservationsList(List<List<float>> list)
    {
        List<float> buffer = new();

        foreach (var b in list)
        {
            foreach (var c in b)
            {
                buffer.Add(c);
            }
        }
        return buffer;
    }
}
