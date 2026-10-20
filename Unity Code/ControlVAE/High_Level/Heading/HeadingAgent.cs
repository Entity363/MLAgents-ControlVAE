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

public class HeadingAgent : ControlVAEAgent
{
    protected readonly Vector3 direction = Vector3.forward;
    protected float dirX = 0f;
    protected float dirY = 0f;
    protected float angle = 0f;
    protected float speed = 0f;
    protected List<float> speeds = new List<float>();
    protected List<float> targetHeading = new List<float>();
    
    public ControlVAECamera _camera;

    public override void Initialize()
    {
        base.Initialize();
        
        speeds = new List<float>{_camera.walkSpeed, _camera.runSpeed, 0};
        targetHeading = RandomTarget(speeds);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        base.OnActionReceived(actionBuffers);

        if (step % 96 == 0 || step == 0)
        {
            targetHeading = RandomTarget(speeds);
        }
    }
    
    //set it to 0 so it doesn't go to weird spawns
    public void UpdateP(int[] init_index)
    {
        base.UpdateP(init_index);
        for (int i = 0; i < p_init.Length; i++)
        {
            p_init[i] = 0;
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        _obsBuffer.Clear();
        //states
        var state = _controller.GetStateBase();
        _obsBuffer.AddRange(state);
        //normalized obs
        var lobs = state2ob(state);
        var obs = Straighten2List(lobs);
        obs = normalize(obs, means.ToList(), stds.ToList());
        _obsBuffer.AddRange(obs);
        //targets
        var target = _controller.GetStateTarget();
        _obsBuffer.AddRange(target);
        //normalized obs target
        var ltobs = state2ob(target);
        var tobs = Straighten2List(ltobs);
        tobs = normalize(tobs, means.ToList(), stds.ToList());
        _obsBuffer.AddRange(tobs);
        //done
        _obsBuffer.Add(doneStep);
        //we want prior = free movement from distribution
        _obsBuffer.Add(PriorVPost);
        

        if (Training)
        {
            speed = targetHeading[0];
            angle = targetHeading[1];
        }
        else
        {
            dirX = _camera.moveCos;
            dirY = _camera.moveSin;
            angle = _camera.transform.eulerAngles.y;
            speed = _camera.moveSpeed;
        }
        //heading is speed, angle
        _obsBuffer.Add(speed);
        _obsBuffer.Add(angle);
        
        foreach (var v in _obsBuffer) sensor.AddObservation(v);
    }
}
