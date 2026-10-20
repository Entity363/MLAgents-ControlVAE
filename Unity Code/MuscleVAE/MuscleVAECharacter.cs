using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.InteropServices;
using Kinesis.Components;
using Kinesis.Core;
using NUnit.Framework;
using Unity.MLAgents.Actuators;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;
using static MuscleVAE.Utils.MotionUtils;

public class MuscleVAECharacter
{
    
    #region  caches

    protected List<MuscleTendonUnit> muscleCache = new();
    protected List<MuscleStimulator> stimulatorsCache = new();
    protected List<float> muscleLengthsCache = new();
    protected List<float> muscleLastLengthsCache = new();
    protected List<float> muscleVelocitiesCache = new();
    protected List<Rigidbody> bodiesCache = new();
    protected List<Vector3> posCache = new();
    protected List<Vector3> lastPosCache = new();
    protected List<Quaternion> rotCache = new();
    protected List<Quaternion> lastRotCache = new();
    protected List<Vector3> velCache = new();
    protected List<Vector3> aVelCache = new();
    
    protected Transform root = null;
    protected Transform head = null;
    
    #endregion

    public void BuildCaches(List<Transform> bodies, List<MuscleTendonUnit> muscles ,Transform lroot, Transform lhead)
    {
        muscleCache.Clear();
        stimulatorsCache.Clear();
        muscleLengthsCache.Clear();
        muscleLastLengthsCache.Clear();
        muscleVelocitiesCache.Clear();
        bodiesCache.Clear();
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
            bodiesCache.Add(body.GetComponent<Rigidbody>());
            posCache.Add(body.position);
            lastPosCache.Add(body.position);
            rotCache.Add(body.rotation);
            lastRotCache.Add(body.rotation);
            
            velCache.Add(Vector3.zero);
            aVelCache.Add(Vector3.zero);
        }
        
        int muscleCount = muscles.Count;
        //Debug.Log("MuscleCount: " + muscleCount);
        for (int i = 0; i < muscleCount; i++)
        {
            var muscle =  muscles[i];
            muscleCache.Add(muscle);
            muscleLengthsCache.Add(muscle.MuscleLength());
            muscleVelocitiesCache.Add(muscle.GetMuscleVelocity());
            var stimulator = muscle.GetComponent<MuscleStimulator>();
            stimulatorsCache.Add(stimulator);
        }

        root = lroot;
        head = lhead;
        
        Debug.Log($"Created caches for {count} bodies, {posCache.Count} positions, {rotCache.Count} rotations, {velCache.Count} velocitys, {aVelCache.Count} avels, {muscleCache.Count} muscles.");
    }

    public void UpdateCaches(List<Transform> bodies, List<MuscleTendonUnit> muscles ,Transform lroot, Transform lhead)
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
        
        int muscleCount = muscles.Count;
        for (int i = 0; i < muscleCount; i++)
        {
            var muscle =  muscles[i];
            var len = muscle.MuscleLength();
            muscleLengthsCache[i] = len;
         
            var vel = muscle.MuscleVelocity();
            muscleVelocitiesCache[i] = vel;
        }
    }

    public void SetRestLengths(List<float> restLengths)
    {
        for (int i = 0; i < muscleCache.Count; i++)
        {
            muscleCache[i].SetRestLength(restLengths[i]);
        }
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

    public List<float> GetMuscleLengths()
    {
        return muscleLengthsCache;
    }
    public List<float> GetMuscleObs(float[] restLengths, float[] seSlackRestLengths, float[] optimalCeRestLengths)
    {
        float[] muscleLengths = new float[muscleLengthsCache.Count];
        for (int i = 0; i < muscleLengths.Length; i++)
        {
            muscleLengths[i] = (muscleLengthsCache[i] / restLengths[i] - seSlackRestLengths[i]) / optimalCeRestLengths[i];
        }
        return muscleLengths.ToList();
    }

    public List<Vector3> GetOffsets()
    {
        List<Vector3> offsets = new List<Vector3>();
        int count = muscleCache.Count;
        for (int i = 0; i < count; i++)
        {
            var muscle = muscleCache[i];
            for (int j = 0; j < muscle.muscleNodes.Count; j++)
            {
                var node = muscle.muscleNodes[i];
                Vector3 offset = node.offset;
                offsets.Add(offset);
            }
        }
        return offsets;
    }
    public List<int> GetIndices()
    {
        List<int> indices = new List<int>();
        int count = muscleCache.Count;
        for (int i = 0; i < count; i++)
        {
            var muscle = muscleCache[i];
            for (int j = 0; j < muscle.muscleNodes.Count; j++)
            {
                var node = muscle.muscleNodes[i];
                indices.Add(j);
            }
        }
        return indices;
    }
    public List<int> GetMuscleBodies()
    {
        List<int> indices = new List<int>();

        Dictionary<Rigidbody, int> cacheIndex = new Dictionary<Rigidbody, int>();
        for (int i = 0; i < bodiesCache.Count; i++)
        {
            cacheIndex[bodiesCache[i]] = i;
        }

        for (int i = 0; i < muscleCache.Count; i++)
        {
            var muscle = muscleCache[i];

            for (int j = 0; j < muscle.muscleSegments.Count; j++)
            {
                var node = muscle.muscleSegments[j];
                var body = node.head.bone.GetComponent<Rigidbody>();

                if (cacheIndex.TryGetValue(body, out int idx))
                {
                    indices.Add(idx);
                }
                else
                {
                    throw new Exception("Body not found in cache");
                }
            }
            var lastBody = muscle.muscleSegments.Last().tail.bone.GetComponent<Rigidbody>();

            if (cacheIndex.TryGetValue(lastBody, out int lastIdx))
            {
                indices.Add(lastIdx);
            }
            else
            {
                throw new Exception("last index not found in cache");
            }
        }

        return indices;
    }
    
    
    public Dictionary<List<Vector3>, List<int>> GetOffsetsIndicesAndBodies()
    {
        return new Dictionary<List<Vector3>, List<int>>();
    }
    

    public void ResetPhysics(List<float> states)
    {
        int count = bodiesCache.Count;


        for (int i = 0; i < count; i++)
        {
            Vector3 pos = new Vector3(states[i * 13], states[i * 13 + 1], states[i * 13 + 2]);
            Quaternion rot = new Quaternion(states[i * 13 + 3], states[i * 13 + 4], states[i * 13 + 5], states[i * 13 + 6]);
            
            posCache[i] = pos;
            rotCache[i] = rot;
            velCache[i] = Vector3.zero;
            aVelCache[i] = Vector3.zero;
            
            bodiesCache[i].transform.position = pos;
            bodiesCache[i].transform.rotation = rot;
            bodiesCache[i].angularVelocity = new Vector3(0,0,0);
            bodiesCache[i].linearVelocity = new Vector3(0,0,0);
        }

        for (int i = 0; i < muscleCache.Count; i++)
        {
            muscleCache[i].ResetMuscle();
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

    public void SetMuscleTargets(List<float> actions)
    {
        Debug.Assert(actions.Count == muscleLengthsCache.Count);

        for (int i = 0; i < actions.Count; i++)
        {
            //var muscle = muscleCache[i];
            var stimulator = stimulatorsCache[i];
            
            stimulator.excitation = actions[i];
            //Debug.Log($"[SetTargets] Frame={Time.frameCount} stimulator={stimulatorsCache[0].GetInstanceID()} excitation={stimulatorsCache[0].excitation}");

        }
    }

    public void CountActions()
    {
        Debug.Log($"Requires actions:  {muscleCache.Count}");
    }
    
}
