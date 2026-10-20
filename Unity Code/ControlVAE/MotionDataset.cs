using System;
using System.Collections.Generic;
using System.ComponentModel;
using Unity.MLAgents.Actuators;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;


[CreateAssetMenu(menuName="ControlVAE/Motion Dataset", fileName="MotionDataset")]
public class MotionDataset : ScriptableObject
{
    [Serializable]
    public class Stats
    {
        public float[] obs_mean;
        public float[] obs_std;
        public float[] delta_mean;
        public float[] delta_std;

        public float[] GetMean()
        {
            return obs_mean;
        }

        public float[] GetStd()
        {
            return obs_std;
        }

        public float[] GetDeltaMean()
        {
            return delta_mean;
        }

        public float[] GetDeltaStd()
        {
            return delta_std;
        }
    }

    [Serializable]
    public class Frame
    {
        public float[] state;
        public float[] observation;
        public byte done;
        
    }

    public int fps;
    public int stateDim;
    public int obsDim;

    public Stats stats = new Stats();
    public List<Frame> frames = new List<Frame>();

    public int FrameCount => frames?.Count ?? 0;

    // Convenience: dataset[frame].state
    public float[] GetState(int frame) => frames[frame].state;
    public float[] GetObs(int frame) => frames[frame].observation;
    public bool GetDone(int frame) => frames[frame].done != 0;

    public int[] Done()
    {
        int[] done_array = new int[frames.Count];
        for (int i = 0; i < done_array.Length; i++)
        {
            done_array[i] =  frames[i].done;
        }
        return done_array;
    }

    public int[] GetFrameArray()
    {
        int[] frame_array = new int[frames.Count];
        for (int i = 0; i < frame_array.Length; i++)
        {
            frame_array[i] = i;
        }
        return frame_array;
    }
    
}
