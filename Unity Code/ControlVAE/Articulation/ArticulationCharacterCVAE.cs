using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.InteropServices;
using NUnit.Framework;
using Unity.MLAgents.Actuators;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;
using static ControlVAE.Utils.MotionUtils;

public class ArticulationCharacterCVAE
{
    #region  caches

    protected List<ArticulationBody> jointsCache = new();
    protected List<Vector3> posCache = new();
    protected List<Vector3> lastPosCache = new();
    protected List<Quaternion> rotCache = new();
    protected List<Quaternion> lastRotCache = new();
    protected List<Vector3> velCache = new();
    protected List<Vector3> aVelCache = new();
    
    protected Transform root = null;
    protected Transform head = null;
    
    #endregion
    
    public void BuildCaches(List<Transform> bodies, Transform lroot, Transform lhead)
    {
        jointsCache.Clear();
        posCache.Clear();
        lastPosCache.Clear();
        rotCache.Clear();
        lastRotCache.Clear();
        velCache.Clear();
        aVelCache.Clear();
        
        int count = bodies.Count;
        for (int i = 0; i < count; i++)
        {
            var body = bodies[i];
            var joint =  body.GetComponent<ArticulationBody>();
            jointsCache.Add(joint);
            
            posCache.Add(body.position);
            lastPosCache.Add(body.position);
            rotCache.Add(body.rotation);
            lastRotCache.Add(body.rotation);
            
            velCache.Add(Vector3.zero);
            aVelCache.Add(Vector3.zero);
        }

        root = lroot;
        head = lhead;
        
        Debug.Log($"Created caches for {count} bodies, {posCache.Count} positions, {rotCache.Count} rotations, {velCache.Count} velocitys, {aVelCache.Count} avels, {jointsCache.Count} joints.");
    }

    public void UpdateCaches(List<Transform> bodies, Transform lroot, Transform lhead)
    {
        int count = bodies.Count;
        for (int i = 0; i < count; i++)
        {
            var pos = bodies[i].position;
            var rot = bodies[i].rotation;
            posCache[i] =  pos;
            rotCache[i] =  rot;
            
            velCache[i] =  GetNumericalVelocity(posCache[i], lastPosCache[i], Time.fixedDeltaTime);
            aVelCache[i] = GetNumericalAngularVelocity(rotCache[i], lastRotCache[i], Time.fixedDeltaTime);
            lastPosCache[i] = pos;
            lastRotCache[i] = rot;
        }
        root = lroot;
        head = lhead;
    }

    public List<float> GetStates()
    {
        List<float> cache = new List<float>();
        
        Debug.Assert(posCache.Count == rotCache.Count );
        Debug.Assert(velCache.Count == aVelCache.Count);
        Debug.Assert(posCache.Count == velCache.Count );

        int count = posCache.Count;
        //Debug.Log($"Getting {count} bodies");
        for (int i = 0; i < count; i++)
        {
            var p = posCache[i];
            var r =  rotCache[i];
            var v = velCache[i];
            var a = aVelCache[i];
            
            cache.Add(p[0]); cache.Add(p[1]); cache.Add(p[2]);
            cache.Add(r[0]); cache.Add(r[1]); cache.Add(r[2]); cache.Add(r[3]);
            cache.Add(v[0]); cache.Add(v[1]); cache.Add(v[2]);
            cache.Add(a[0]); cache.Add(a[1]); cache.Add(a[2]);
        }
        
        return cache;
    }

    public List<float> GetJointTargets()
    {
        List<float> cache = new List<float>();
        jointsCache[0].GetJointPositions(cache);
        return cache;
    }

    public void SetJointPositions(List<float> states, Vector3 rootPos, Quaternion rootRot)
    {
        jointsCache[0].TeleportRoot(rootPos, rootRot);
        jointsCache[0].SetJointPositions(states);
        for (int i = 0; i < jointsCache.Count; i++)
        {
            jointsCache[i].angularVelocity = new Vector3(0,0,0);
            jointsCache[i].linearVelocity = new Vector3(0,0,0);
        }
    }

    public void SetBodiesPositions(List<float> states)
    {
        for (int i = 0; i < jointsCache.Count; i++)
        {
            jointsCache[i].Sleep();
            jointsCache[i].enabled = false;
        }
        for (int i = 0; i < jointsCache.Count; i++)
        {
            Vector3 pos = new Vector3(states[i * 13], states[i * 13 + 1], states[i * 13 + 2]);
            Quaternion rot = new Quaternion(states[i * 13 + 3], states[i * 13 + 4], states[i * 13 + 5], states[i * 13 + 6]);
            
            posCache[i] = pos;
            rotCache[i] = rot;
            velCache[i] = Vector3.zero;
            aVelCache[i] = Vector3.zero;
            
            jointsCache[i].transform.position = pos;
            jointsCache[i].transform.rotation = rot;
            jointsCache[i].angularVelocity = new Vector3(0,0,0);
            jointsCache[i].linearVelocity = new Vector3(0,0,0);
        }
        for (int i = 0; i < jointsCache.Count; i++)
        {
            jointsCache[i].enabled = true;
            jointsCache[i].WakeUp();
        }
    }

    public void SetPose(float[] states)
    {
        Debug.Assert(states.Length % 13 == 0);
        Debug.Assert(posCache.Count == rotCache.Count );
        Debug.Assert(velCache.Count == aVelCache.Count);
        Debug.Assert(posCache.Count == velCache.Count );
        
        for (int i = 0; i < posCache.Count; i++)
        {
            Vector3 pos = new Vector3(states[i * 13], states[i * 13 + 1], states[i * 13 + 2]);
            Quaternion rot = new Quaternion(states[i * 13 + 3], states[i * 13 + 4], states[i * 13 + 5], states[i * 13 + 6]);
            
            posCache[i] = pos;
            rotCache[i] = rot;
            velCache[i] = Vector3.zero;
            aVelCache[i] = Vector3.zero;
        }
    }

    public void SetArticulationFromTarget(Transform[] states)
    {
        List<float> list = new List<float>();
        for (int i = 0; i < states.Length; i++)
        {
            var body = jointsCache[i];
            if (body.jointType == ArticulationJointType.SphericalJoint)
            {
                List<float> l = new List<float>();
                var s = states[i].transform.localRotation;
                l.Add(s[0]); l.Add(s[1]); l.Add(s[2]); l.Add(s[3]);
                list.AddRange(l);
            }

            if (body.jointType == ArticulationJointType.RevoluteJoint)
            {
                var s = states[i].transform.localRotation.x;
                list.Add(s);
            }
        }
        jointsCache[0].SetJointPositions(list);
    }
    
    public void SetPoseWithArticulation(float[] states)
    {
        Debug.Assert(states.Length % 13 == 0);
        Debug.Assert(states.Length / 13 == jointsCache.Count, "NUMBER OF ARTICULATION BODIES DOESN'T MATCH NUMBER OF BODIES IN STATES");

        for (int i = 0; i < jointsCache.Count; i++)
        {
            //jointsCache[i].enabled = false;
        }

        Vector3 rpos = new Vector3(states[0], states[1], states[2]);
        Quaternion rrot = new Quaternion(states[3], states[4], states[5], states[6]);
        jointsCache[0].TeleportRoot(rpos, rrot);
        for (int i = 0; i < jointsCache.Count; i++)
        {
            ArticulationBody body = jointsCache[i];
            Vector3 pos = new Vector3(states[i * 13], states[i * 13 + 1], states[i * 13 + 2]);
            Quaternion rot = new Quaternion(states[i * 13 + 3], states[i * 13 + 4], states[i * 13 + 5], states[i * 13 + 6]);
            
            //body.transform.position = pos;
            //body.transform.rotation = rot;
            
            
            //if (body.isRoot) body.TeleportRoot(pos, rot);

            //body.jointPosition = new ArticulationReducedSpace(pos.x, pos.y, pos.z);
            //body.jointForce  = new ArticulationReducedSpace(0f, 0f, 0f);
            //body.jointVelocity = new ArticulationReducedSpace(0f, 0f, 0f);
            
            List<float> tmp = new List<float>();
            int totalDOF = body.GetJointPositions(tmp);
            //Debug.Log(tmp.Count);
            int k = 0;
            for(int j = 0; j < tmp.Count; j++)
                tmp[j] = 0;
            body.SetJointPositions(tmp);
            //body.ResetInertiaTensor();
            //body.velocity = Vector3.zero;
            //body.angularVelocity = Vector3.zero;
            
            
            posCache[i] = pos;
            rotCache[i] = rot;
            velCache[i] = Vector3.zero;
            aVelCache[i] = Vector3.zero;
            
        }
        for (int i = 0; i < jointsCache.Count; i++)
        {
            //jointsCache[i].enabled = true;
        }
    }
    
    public void SetTargets(List<float> actions)
    {
        int k = 0;
        int count = jointsCache.Count;
        for (int i = 0; i < count; i++)
        {
            ArticulationBody body = jointsCache[i];
            if (body.jointType == ArticulationJointType.RevoluteJoint)
            {
                var x = actions[k];
                var y = actions[k + 1];
                var z = actions[k + 2];
                
                x = (x + 1) * 0.5f;
                //y = (y + 1) * 0.5f;
                //z = (z + 1) * 0.5f;
                
                var xRot = Mathf.Lerp(body.xDrive.lowerLimit, body.xDrive.upperLimit, x);
                
                body.SetDriveTarget(ArticulationDriveAxis.X, xRot);
                k += 3;
            }
            else if (body.jointType == ArticulationJointType.SphericalJoint)
            {
                var x = actions[k];
                var y = actions[k + 1];
                var z = actions[k + 2];
                
                x = (x + 1) * 0.5f;
                y = (y + 1) * 0.5f;
                z = (z + 1) * 0.5f;
                
                var xRot = Mathf.Lerp(body.xDrive.lowerLimit, body.xDrive.upperLimit, x);
                var yRot = Mathf.Lerp(body.yDrive.lowerLimit, body.yDrive.upperLimit, y);
                var zRot = Mathf.Lerp(body.zDrive.lowerLimit, body.zDrive.upperLimit, z);
                
                body.SetDriveTarget(ArticulationDriveAxis.X, xRot);
                body.SetDriveTarget(ArticulationDriveAxis.Y, yRot);
                body.SetDriveTarget(ArticulationDriveAxis.Z, zRot);
                k += 3;
            }
            else if (body.jointType == ArticulationJointType.FixedJoint)
            {
                continue;
            }
        }
    }

    public void CountActions()
    {
        int k = 0;
        int count = jointsCache.Count;
        for (int i = 0; i < count; i++)
        {
            ArticulationBody body = jointsCache[i];
            if (body.jointType == ArticulationJointType.RevoluteJoint)
            {
                //k++;
                k+= 3;
            }
            else if (body.jointType == ArticulationJointType.SphericalJoint)
            {
                k += 3;
            }
            else if (body.jointType == ArticulationJointType.FixedJoint)
            {
                continue;
            }
        }
        Debug.Log($"_joints.Count = {jointsCache.Count}, expected actions = {k}");
    }
}
