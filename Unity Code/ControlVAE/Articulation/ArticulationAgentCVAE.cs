using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.SideChannels;
using static ControlVAE.Utils.MotionUtils;
using static ControlVAE.Utils.IndexCounter;

public class ArticulationAgentCVAE : Agent
{
    [Header("General")]
    [SerializeField] protected ArticulationControllerCVAE _controller;
    [SerializeField] protected MotionDataset motionData;
    protected FloatArraySideChannel _sideChannel;
    
    [Header("Episode settings")]
    public float fallHeightThreshold = 0.6f;
    public float fallPenality = -0.00f;  //from deepmimic, ignore
    protected int step = 0;
    protected int doneStep = 0;
    public int maxSteps = 512;
    public int minStep = 26;
    
    [Header("Prior vs Post")] public float PriorVPost = 1;
    
    #region cache
    
    protected float[] val;
    protected float[] p_init;
    
    protected readonly List<float> _obsBuffer = new();
    protected readonly List<float> _actionBuffer = new();

    protected int i = 0;
    protected int[] init_index;
    protected readonly List<int> epFrameNum = new();
    protected readonly List<float> epRwd = new();
    protected readonly List<int> epDone = new();

    protected int startFrame;

    protected float[] means;
    protected float[] stds;
    protected float[] delta_means;
    protected float[] delta_stds;
    
    protected int fallenStepsInRow = 0;
    
    protected List<int> epsSize = new();
    protected int epsSizeAvg = 0;
    
    #endregion

    public override void Initialize()
    {
        Debug.Assert(_controller != null, "No DeepMimic controller found");
        int n = motionData.frames.Count;
        val = new float[n];
        
        init_index = CalculateFeasibleIndex(motionData.Done(), 2); 
        UpdateP(init_index);

        //send stats for training policy and wm
        _sideChannel = new FloatArraySideChannel();

        means = motionData.stats.GetMean();
        stds = motionData.stats.GetStd();
        delta_means = motionData.stats.GetDeltaMean();
        delta_stds = motionData.stats.GetDeltaStd();
        
        _sideChannel.SendToPython(means, "obs_mean");
        _sideChannel.SendToPython(stds, "obs_std");
        _sideChannel.SendToPython(delta_means, "delta_mean");
        _sideChannel.SendToPython(delta_stds, "delta_std");
        
        SideChannelManager.RegisterSideChannel(_sideChannel);
    }
    
    private void OnDestroy()
    {
        if (Academy.IsInitialized)
        {
            SideChannelManager.UnregisterSideChannel(_sideChannel);
        }
    }
    public override void OnEpisodeBegin()
    {
        epFrameNum.Clear();
        epRwd.Clear();
        epDone.Clear();
        epsSize.Clear();

        // update p per iteration
        UpdateP(init_index);
        
        Reset();
        
    }
    
    //Instead of ending episode on each trajectory, let it end at the max rollout length, and append trajectories with done as obs
    public void Reset()
    {
        startFrame = RandomSelect(init_index, p_init);
        i = startFrame;
        _controller.SetPose(_controller._target,motionData.frames[i].state);
        
        //_controller.SetAnimPhase(0);
        _controller.SetPose(_controller._base,motionData.frames[i].state);
        step = 0;
    }
    
    public void FixedUpdate()
    {
        _controller.SetPose(_controller._target,motionData.frames[i].state);
        
    }
    
    
    //formulation is states base + normalized obs base + states target + normalized obs target + done flag + prior vs posterior bool
    public override void CollectObservations(VectorSensor sensor)
    {
        var statesBase = _controller._base.GetStates();
        var obsBase = state2ob(statesBase);
        var lobsBase = Straighten2List(obsBase);
        var nobsBase = normalize(lobsBase, means.ToList(), stds.ToList());
        
        var statesTarget = _controller._target.GetStates();
        var obsTarget = state2ob(statesTarget);
        var lobsTarget = Straighten2List(obsTarget);
        var nobsTarget = normalize(lobsTarget, means.ToList(), stds.ToList());
        
        sensor.AddObservation(statesBase);
        sensor.AddObservation(nobsBase);
        sensor.AddObservation(statesTarget);
        sensor.AddObservation(nobsTarget);
        sensor.AddObservation(doneStep);
        sensor.AddObservation(PriorVPost);
        
        epRwd.Add(_controller.CallRwd(lobsBase, lobsTarget, Time.fixedDeltaTime));
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        _actionBuffer.Clear();
        
        var actions = actionBuffers.ContinuousActions;
        for (int i = 0; i < actions.Length; i++)
        {
            _actionBuffer.Add(actions[i]);
        }
        if (actions.Length > 0)
        {
            _controller.SetTargets(actionBuffers);
        }
        else
        {
            //_controller.SetJointTargetRotationWithMaxForce(default);
        }
        
        //not actual agent reward, only for index biasing
        //epRwd.Add(_controller.GetRwd());

        bool reset = false;
        doneStep = 0;

        if (true)
        {
            //if fallen
            if (_controller.IsFallen(fallHeightThreshold))
            {
                fallenStepsInRow++;
            }
            else
            {
                fallenStepsInRow = 0;
            }

            //reset only if fallen is true for consecutive frames
            if (fallenStepsInRow >= minStep)
            {
                doneStep = 1;
                reset = true;
                fallenStepsInRow = 0;
            }
            //if ran out of frames
            else if (i >= motionData.frames.Count - 2)
            {
                doneStep = 2;
                reset = true;
            }
            //if exceded 512 risk instability in replay buffer
            else if (step > maxSteps)
            {
                doneStep = 3;
                reset = true;
            }
        }


        //first append donestep for obs, then step only if it's not resetting, otherwise first frame is skipped
        epDone.Add(doneStep);
        if (reset == false)
        {
            step++;
            i++;
        }
        else
        {
            epsSize.Add(step);
            AddReward(fallPenality);
            Reset();
            //update statistics
            epsSizeAvg = GetAvg(epsSize);
            var stats = Academy.Instance.StatsRecorder;
            stats.Add("Episode/episode length: ", epsSizeAvg);
            
            float mean = epRwd.Average();
            float std = MathF.Sqrt(epRwd.Average(v => (v - mean) * (v - mean)));
            stats.Add("Episode/rwd mean: ", mean);
            stats.Add("Episode/rwd std: ", std);

        }
    }
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var a = actionsOut.ContinuousActions;
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = UnityEngine.Random.Range(-1f, 1f);
        }
    }




    //temperature based on error
    public void UpdateVal(int[] done, float[] rwd, int[] frame_num)
    {
        float[] tmp_val = new float[val.Length];
        for (int i = 0; i < val.Length; i++)
        {
            tmp_val[i] = val[i] / 2;
        }
        int last_i = 0;

        for (int i = 0; i < frame_num.Length; i++)
        {
            if (done[i] != 0)
            {
                if (done[i] == 1)
                {
                    tmp_val[frame_num[i]] = rwd[i];
                }
                else
                {
                    tmp_val[frame_num[i]] = 0;
                }

                for (int j = i - 1; j >= last_i; j--)
                {
                    tmp_val[frame_num[j]] = 0.95f * tmp_val[frame_num[j+1]] + rwd[j];
                }
                last_i = i;
            }
        }

        for (int i = 0; i < val.Length; i++)
        {
            val[i] = 0.9f * val[i] + 0.1f * tmp_val[i];
        }
    }
    //probability distribution of indices based on error
    public void UpdateP(int[] init_index)
    {
        float[] p = new float[val.Length];
        for (int i = 0; i < val.Length; i++)
        {
            p[i] = 1f / Mathf.Max(0.01f, val[i]);
        }

        p_init = new float[init_index.Length];

        float sum = 0f;
        for (int i = 0; i < init_index.Length; i++)
        {
            int idx = init_index[i];
            float w;
            if (idx >= 0 && idx < p.Length)
            {
                w = p[idx];
            }
            else
            {
                w = 0f;
            }
            p_init[i] = w;
            sum += w;
        }

        if (sum > 0.1f)
        {
            float inv = 1f / sum;
            for (int i = 0; i < p_init.Length; i++)
            {
                p_init[i] *= inv;
            }
        }
    }
}
