using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class DeepMimicAgent : Agent
{
    [Header("General")]
    [SerializeField] private DeepMimicController _controller;
    
    [Header("Reward settings")]
    public float wAlive = 0.20f;

    public float qScale = 2f;
    public float vScale = 0.1f;
    public float comScale = 10f;
    public float eeScale = 40f;

    [Header("Weights")]
    public float w_I = .7f;

    public float weightPos = .65f;
    public float weightVel = .1f;
    public float weightCom = .15f;
    public float weightEE = .1f;

    // torque penalty scale
    public float wTau = 1e-6f;

    [Header("Heading reward")]
    public float w_g = 0.3f;

    public float desiredSpeed = 1.5f;
    public Vector3 targetDirWorld = Vector3.forward;
    public float headingScale = 2.5f;

    [Header("Episode settings")]
    public float maxEpisodeTime = 10f;
    public float fallHeightThreshold = 0.6f;
    public float fallPenality = -0.10f;

    #region buffer

    private float _episodeTime = 0;
    private float _minEpisodeTime = 0.1f;
    
    
    private readonly List<float> _obsBuffer = new();
    private readonly List<float> _actionBuffer = new();

    #endregion

    public override void Initialize()
    {
        Debug.Assert(_controller != null, "No DeepMimic controller found");
        
    }

    public override void OnEpisodeBegin()
    {
        _controller.SetAnimPhase(0);
        _controller.ResetPhysicsCharacter();
        _episodeTime = 0;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        _obsBuffer.Clear();
        _controller.GetObservations(_obsBuffer);
        foreach (var v in _obsBuffer) sensor.AddObservation(v);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        _episodeTime += Time.fixedDeltaTime;
        _actionBuffer.Clear();
        
        var actions = actionBuffers.ContinuousActions;
        for (int i = 0; i < actions.Length; i++)
        {
            _actionBuffer.Add(actions[i]);
        }
        if (actions.Length > 0)
        {
            _controller.SetJointTargets(_actionBuffer);
        }
        else
        {
            //_controller.SetJointTargets(default);
        }
        if (_controller.IsFallen(fallHeightThreshold) && _episodeTime > _minEpisodeTime)
        {
            AddReward(fallPenality);
            EndEpisode();
            return;
        }
        float reward = ComputeReward();
        AddReward(reward );
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var a = actionsOut.ContinuousActions;
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = UnityEngine.Random.Range(-1f, 1f);
        }
    }

    public float ComputeReward()
    {
        float reward = 0.0f;
        var stats = Academy.Instance.StatsRecorder;

        //imitation
        float imitationRwd = 0;
        float posRwd = weightPos * Mathf.Exp(-_controller.GetRotDiffSqr() * qScale);
        float velRwd = weightVel * Mathf.Exp(-_controller.GetVelDiffSqr() * vScale);
        float eeRwd = weightEE * Mathf.Exp(-_controller.GetEeDiffSqr() * eeScale);
        float comRwd = weightCom * Mathf.Exp(-_controller.GetComDiffSqr() * comScale);
        
        imitationRwd = posRwd + velRwd + eeRwd + comRwd;
        
        //task specific
        //target heading
        float targetRwd = Mathf.Exp(-_controller.GetHeadingRwd(desiredSpeed) * headingScale);
        
        //sum = 1
        reward = w_I * imitationRwd + w_g * targetRwd;
        
        stats.Add("Humanoid/r_p",posRwd);
        stats.Add("Humanoid/r_v",velRwd);
        stats.Add("Humanoid/r_c",comRwd);
        stats.Add("Humanoid/r_e",eeRwd);
        stats.Add("Humanoid/r_t", imitationRwd);
        stats.Add("Task/r_g", targetRwd);
        
        return reward;
    }
    
    
    
}
