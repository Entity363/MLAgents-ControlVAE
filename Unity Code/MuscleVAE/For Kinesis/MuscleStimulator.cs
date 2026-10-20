using Kinesis.Core;
using System.Collections.Generic;
using UnityEngine;


namespace Kinesis.Components {
    /// <summary>
    /// Component for testing individual muscles from the Unity interface.
    /// </summary>
    [RequireComponent(typeof(MuscleTendonUnit))]
    [DisallowMultipleComponent]
    public class MuscleStimulator : MonoBehaviour {
        public const float MinExcitation = -1.0f;
        public const float MaxExcitation = 1.0f;
        public bool runFixedUpdate;
        //[Range(MinExcitation, MaxExcitation)]
        public float excitation;
        public MuscleTendonUnit muscle;
        public bool debug = false;
        private Dictionary<MuscleSegment, Vector3> jointTorques;

        void Awake() {
            this.jointTorques = new Dictionary<MuscleSegment, Vector3>();
        }

        void FixedUpdate() {
            if (!runFixedUpdate) {
                return;
            }
            if (debug) {
                //Debug.Log($"[FixedUpdate] Frame={Time.frameCount} this={GetInstanceID()} excitation={excitation}");
                //Debug.Log($"[Stimulator] {gameObject.name} excitation={excitation} muscle={muscle != null}");
            }
            //this.muscle.CalculateJointTorques(this.excitation, ref this.jointTorques);
            //this.muscle.CalculatePDTorques(this.excitation, ref this.jointTorques);
            this.muscle.CalculatePDTorquesMVAE(this.excitation, ref this.jointTorques);
            if (debug)
            {
                //Debug.Log($"[Stimulator] {gameObject.name} excitation={excitation} muscle={muscle != null} ");
                //Debug.Log($"[Stimulator] calling CalculatePDTorques with excitation={excitation} on muscle instanceID={muscle.GetInstanceID()}");

            }
            MuscleTendonUnit.ApplyJointTorques(this.jointTorques);
            //MuscleTendonUnit.ApplyJointForces(this.jointTorques);
        }

        void Reset() {
            this.muscle = this.gameObject.GetComponent<MuscleTendonUnit>();
        }
    }
}