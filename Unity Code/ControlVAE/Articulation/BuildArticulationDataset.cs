using UnityEngine;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using Unity.MLAgents.Actuators;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;
using static ControlVAE.Utils.MotionUtils;
using static ControlVAE.Utils.IndexCounter;

public class BuildArticulationDataset : MonoBehaviour
{
    private List<float> obs_mean = new List<float>();
    private List<float> obs_std = new List<float>();
    
    private List<float> delta_mean = new List<float>();
    private List<float> delta_std = new List<float>(); 
    
    private readonly List<List<float>> buffers = new();
    private readonly List<List<float>> states = new();
    private readonly List<float> done = new();
    private readonly List<List<float>> jointTargets = new();
    
    [Header("Motion")]
    public List<AnimationClip> animationClips = new();
    private int numAnims = 0;
    private int animCount = 0;
    private bool armed = true;
    //public Animator animator;
    
    public ArticulationControllerCVAE controlVAEController; //must be before motiondataset otherwise it compiles weird
    public MotionDataset output;
    

    
    [Header("Timestep")]
    public float scale = 10;
    public int fps = 50;
    
    private float sampleDt = 1f / 50;
    
    private float accum = 0f;
    private bool stopping = false;
    
    void Start()
    {
        Time.timeScale = scale;
        //Time.fixedDeltaTime = 0.02f / scale;  
        
        buffers.Clear();
        states.Clear();
        done.Clear();

        controlVAEController.SetAnimPhase(0);
        
        
        
        // from google
        AnimationClip[] clips = controlVAEController.animator.runtimeAnimatorController.animationClips;
        numAnims = clips.Length;
        Debug.Log("Total number of animation clips in the Animator: " + numAnims);
        foreach (AnimationClip clip in clips)
        {
            Debug.Log("Clip Name: " + clip.name);
            continue;
        }
        
        
    }

    void FixedUpdate()
    {
        var state = controlVAEController._target.GetStates();
        states.Add(new List<float>(state));
        var ltobs = state2ob(state);
        var tobs = Straighten2List(ltobs);
        buffers.Add(new List<float>(tobs));

        var tgt = controlVAEController._target.GetJointTargets();
        jointTargets.Add(tgt);

        
        float phase = controlVAEController.GetAnimPhase();

        if (armed && phase >= 0.999f)
        {
            animCount += 1;
            armed = false;
        }

        if (phase < 0.5f)
        {
            armed = true;
        }
        if (animCount >= numAnims)
        {
            done.Add(1);
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
        Application.Quit();
#endif
        }
        else
        {
            done.Add(0);
        }
    }

    void OnDestroy()
    {
        Debug.Log($"frames = {buffers.Count}");
        
        ComputeStatistics();

        int body = 0;
        int body0Start = body + 0;
        int body0Count = body + 16;

        var obsbody0 = obs_mean.GetRange(body0Start, body0Count);
        Debug.Log("mean body 0 = " + string.Join(", ", obsbody0));
        var stdbody0 = obs_std.GetRange(body0Start, body0Count);
        Debug.Log("std body 0 = " + string.Join(", ", stdbody0));
        var deltabody0 = delta_mean.GetRange(body0Start, body0Count);
        Debug.Log("delta mean body 0 = " + string.Join(", ", deltabody0));
        var deltastdbody0 = delta_std.GetRange(body0Start, body0Count);
        Debug.Log("delta std body 0 = " + string.Join(", ", deltastdbody0));
        
        
        Time.timeScale = 1f;
        Time.fixedDeltaTime = 0.02f;
        
        WriteToAsset();
    }
    
    
    void WriteToAsset()
    {
        if (output == null)
        {
            Debug.LogError("No MotionDataset asset assigned.");
            return;
        }

        int F = buffers.Count;
        if (F == 0) return;

        output.fps = fps;
        output.stateDim = states[0].Count;
        output.obsDim = buffers[0].Count;

        output.stats.obs_mean = obs_mean.ToArray();
        output.stats.obs_std = obs_std.ToArray();
        output.stats.delta_mean = delta_mean.ToArray();
        output.stats.delta_std = delta_std.ToArray();

        output.frames.Clear();
        output.frames.Capacity = F;

        for (int i = 0; i < F; i++)
        {
            output.frames.Add(new MotionDataset.Frame
            {
                state = states[i].ToArray(),
                observation = buffers[i].ToArray(),                          //jointTargets[i].ToArray(),
                done = (byte)(done[i] != 0f ? 1 : 0),
            });
        }

#if UNITY_EDITOR
        EditorUtility.SetDirty(output);
        AssetDatabase.SaveAssets();
#endif

        Debug.Log($"Wrote dataset frames={output.FrameCount} stateDim={output.stateDim} obsDim={output.obsDim}");
    }

    void ComputeStatistics()
    {
        obs_mean.Clear();
        obs_std.Clear();
        delta_mean.Clear();
        delta_std.Clear();

        int frameCount = buffers.Count;
        if (frameCount < 2)
        {
            Debug.LogError("Need at least 2 frames.");
            return;
        }

        int obsDims = buffers[0].Count;

        // Sanity: all frames same length
        for (int f = 1; f < frameCount; f++)
            if (buffers[f].Count != obsDims)
                Debug.LogError($"Obs dim mismatch at frame {f}: {buffers[f].Count} vs {obsDims}");

        // -------- obs mean/std (Welford) --------
        double[] obsMean = new double[obsDims];
        double[] obsM2   = new double[obsDims];
        int nObs = 0;

        for (int f = 0; f < frameCount; f++)
        {
            nObs++;
            var v = buffers[f];

            for (int i = 0; i < obsDims; i++)
            {
                double x = v[i];
                double d = x - obsMean[i];
                obsMean[i] += d / nObs;
                obsM2[i]   += d * (x - obsMean[i]);
            }
        }

        for (int i = 0; i < obsDims; i++)
        {
            double var = obsM2[i] / frameCount; // numpy ddof=0
            double sd  = Math.Sqrt(Math.Max(0.0, var));
            if (sd < 0.1) sd = 0.1;

            obs_mean.Add((float)obsMean[i]);
            obs_std.Add((float)sd);
        }

        // -------- delta mean/std over vel+avel only --------
        // You must know where vel/avel live inside the FLAT vector.
        // In your per-body layout: [pos0..2, rot6, vel9..11, avel12..14, height15] per body.
        // So for body b, vel starts at base = 16*b + 9, avel at 16*b + 12.
        // updir is at the end and is NOT included in delta stats in the python code.

        int bodyCount = (obsDims - 3) / 16;   // last 3 are updir
        int deltaDims = 6 * bodyCount;

        double[] dMean = new double[deltaDims];
        double[] dM2   = new double[deltaDims];
        int nDelta = 0;

        for (int f = 0; f < frameCount - 1; f++)
        {
            nDelta++;
            var prev = buffers[f];
            var next = buffers[f + 1];

            int idx = 0;
            for (int b = 0; b < bodyCount; b++)
            {
                int baseIdx = 16 * b;

                // vel delta
                for (int k = 0; k < 3; k++)
                {
                    double x = next[baseIdx + 9 + k] - prev[baseIdx + 9 + k];
                    double d = x - dMean[idx];
                    dMean[idx] += d / nDelta;
                    dM2[idx]   += d * (x - dMean[idx]);
                    idx++;
                }

                // avel delta
                for (int k = 0; k < 3; k++)
                {
                    double x = next[baseIdx + 12 + k] - prev[baseIdx + 12 + k];
                    double d = x - dMean[idx];
                    dMean[idx] += d / nDelta;
                    dM2[idx]   += d * (x - dMean[idx]);
                    idx++;
                }
            }
        }

        int deltaCount = frameCount - 1;
        for (int i = 0; i < deltaDims; i++)
        {
            double var = dM2[i] / deltaCount;
            double sd  = Math.Sqrt(Math.Max(0.0, var));
            if (sd < 0.1) sd = 0.1;

            delta_mean.Add((float)dMean[i]);
            delta_std.Add((float)sd);
        }
    }
}
