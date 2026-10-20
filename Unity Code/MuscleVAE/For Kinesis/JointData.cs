using Kinesis.Core;
using System;
using System.Collections.Generic;
using UnityEngine;

namespace Kinesis.Components {
    /// <summary>
    /// Data container for storing additional joint-related data.
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(Joint))]
    [RequireComponent(typeof(Rigidbody))]
    public class JointData : MonoBehaviour {
        [Tooltip("Reference to sibling Joint component this holds data for.")]
        public Joint joint;
        [Tooltip("Reference to sibling anchor Rigidbody component.")]
        public Rigidbody anchorBody;
        [Tooltip("Automatically scan for spanning muscle segments.")]
        public bool autoScan = true;
        [SerializeReference]
        [Tooltip("List of spanning muscle segments.")]
        public List<MuscleSegment> spanningMuscleSegments;

        public void ScanMuscles() {
            if (this.joint == null) {
                string errorMessage = "Missing configured joint.";
                throw new InvalidOperationException(errorMessage);
            }

            this.spanningMuscleSegments.Clear();

            // Find all detectable MTU objects.
            MuscleTendonUnit[] muscles = GameObject.FindObjectsOfType<MuscleTendonUnit>();

            // Scan all muscle segments in all muscles.
            foreach (MuscleTendonUnit muscle in muscles) {
                if (muscle.muscleSegments == null || muscle.muscleSegments.Count < 1) {
                    continue;
                }

                MuscleSegment spanningSegment = muscle.muscleSegments.Find(segment => segment.joint == this.joint);

                if (spanningSegment != null) {
                    this.spanningMuscleSegments.Add(spanningSegment);
                }
            }
        }

        void OnValidate() {
            if (this.autoScan) {
                this.ScanMuscles();
            }
        }

        void Reset() {
            this.joint = this.gameObject.GetComponent<Joint>();
            this.anchorBody = this.gameObject.GetComponent<Rigidbody>();
            this.spanningMuscleSegments = new List<MuscleSegment>();

            this.ScanMuscles();
        }
    }
}