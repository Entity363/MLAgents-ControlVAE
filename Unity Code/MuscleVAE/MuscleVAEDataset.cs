using System;
using System.Collections.Generic;
using System.ComponentModel;
using Unity.MLAgents.Actuators;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;

[CreateAssetMenu(menuName="MuscleVAE/Motion Dataset", fileName="MotionDataset")]
public class MuscleVAEDataset : MotionDataset
{
    public Stats muscleStats = new Stats();
    
    //public Frame muscleFrame = new Frame();
    public List<Frame> muscleFrames = new List<Frame>();
}
