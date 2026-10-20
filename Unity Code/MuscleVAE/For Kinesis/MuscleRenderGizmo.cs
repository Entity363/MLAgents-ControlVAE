using Kinesis.Core;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

#if UNITY_EDITOR
namespace Kinesis.Components {
    /// <summary>
    /// Component for muscle data visualization in Scene view.
    /// </summary>
    [DisallowMultipleComponent]
    public class MuscleRenderGizmo : SingletonMonoBehaviour<MuscleRenderGizmo> {
        [Tooltip("Highlight selected muscle paths in Scene view.")]
        public bool highlightSelected = true;
        [Tooltip("Display muscle paths of nonselected muscles in Scene view.")]
        public bool displayNonselected = true;
        [Tooltip("Display muscle segment lines in Scene view.")]
        public bool displaySegments = true;
        [Tooltip("Display physics lever arms in Scene view.")]
        public bool displayLeverArms = true;
        [Tooltip("Display joint torques in Scene view (during Play Mode).")]
        public bool displayTorqueVectors = true;
        [Tooltip("Color for muscles currently selected in the hierarchy.")]
        public Color highlightColor = Color.green;
        [Tooltip("Color for muscles with no activation.")]
        public Color inactiveMuscleColor = new Color(0.35f, 0.0f, 0.0f, 0.8f);
        [Tooltip("Color for muscles with full activation.")]
        public Color activeMuscleColor = new Color(1.0f, 0.0f, 0.0f, 1.0f);
        [Tooltip("Color for lever arm lines.")]
        public Color leverArmColor = new Color(0.5f, 0.5f, 0.5f, 0.67f);
        [Tooltip("Color for joint torque rays.")]
        public Color torqueVectorColor = Color.cyan;
        [Tooltip("Torque vector visualization scaling.")]
        public float torqueScaling = 0.01f;
        public float highlightedNodeRadius = 0.008f;
        public float nonselectedNodeRadius = 0.004f;
        public Dictionary<MuscleTendonUnit, Dictionary<MuscleSegment, Vector3>> muscleJointTorquesMap;
        [SerializeField]
        private List<MuscleTendonUnit> registeredMuscles;

        public void ScanMuscles() {
            MuscleTendonUnit[] muscles = GameObject.FindObjectsOfType<MuscleTendonUnit>();
            this.registeredMuscles.Clear();

            foreach (MuscleTendonUnit muscle in muscles) {
                this.registeredMuscles.Add(muscle);
            }
        }

        void RegisterScanCallback() {
            // Protect callback from being hooked multiple times.
            EditorApplication.hierarchyChanged -= this.ScanMuscles;
            // Register ScanMuscles to be called when the hierarchy changes.
            EditorApplication.hierarchyChanged += this.ScanMuscles;
        }

        void RenderMusclePath(MuscleTendonUnit muscle, Color color, float radius) {
            Gizmos.color = color;
            MuscleNode prevNode = null;
            foreach (MuscleNode node in muscle.muscleNodes) {
                if (node.bone == null) {
                    continue;
                }

                Vector3 nodePosition = node.bone.transform.TransformPoint(node.offset);
                Gizmos.DrawWireSphere(nodePosition, radius);

                if (this.displaySegments && prevNode != null) {
                    Vector3 prevNodePosition = prevNode.bone.transform.TransformPoint(prevNode.offset);
                    Gizmos.DrawLine(prevNodePosition, nodePosition);
                }
                prevNode = node;
            }
        }

        void RenderLeverArms(MuscleTendonUnit muscle, Color color) {
            foreach (MuscleSegment segment in muscle.muscleSegments) {
                if (segment.joint == null) {
                    continue;
                }

                Vector3 jointPosition = segment.joint.transform.TransformPoint(segment.joint.anchor);
                Vector3 tailPosition = segment.tail.bone.transform.TransformPoint(segment.tail.offset);

                Gizmos.color = color;
                Gizmos.DrawLine(jointPosition, tailPosition);
            }
        }

        void RenderJointTorques(Dictionary<MuscleSegment, Vector3> jointTorques, Color color) {
            foreach (KeyValuePair<MuscleSegment, Vector3> kvp in jointTorques) {
                MuscleSegment segment = kvp.Key;
                Vector3 jointTorque = kvp.Value;
                Vector3 jointPosition = segment.joint.transform.TransformPoint(segment.joint.anchor);

                Gizmos.color = color;
                Gizmos.DrawRay(jointPosition, jointTorque * this.torqueScaling);
            }
        }

        void Awake() {
            this.muscleJointTorquesMap = new Dictionary<MuscleTendonUnit, Dictionary<MuscleSegment, Vector3>>();
        }

        void OnDrawGizmos() {
            if (this?.registeredMuscles == null) {
                return;
            }

            foreach (MuscleTendonUnit muscle in this.registeredMuscles) {
                if (muscle == null) {
                    Debug.LogError("Null muscle");
                    continue;
                }

                bool isSelected = Selection.Contains(muscle.gameObject);

                if (!this.highlightSelected || !isSelected) {
                    Color muscleColor = Color.Lerp(
                        this.inactiveMuscleColor,
                        this.activeMuscleColor,
                        muscle.activation
                    );

                    if (isSelected) {
                        this.RenderMusclePath(muscle, muscleColor, this.nonselectedNodeRadius);
                    } else if (this.displayNonselected) {
                        this.RenderMusclePath(muscle, muscleColor, this.nonselectedNodeRadius);
                    }
                } else {
                    this.RenderMusclePath(muscle, this.highlightColor, this.highlightedNodeRadius);
                }

                if (this.displayLeverArms) {
                    this.RenderLeverArms(muscle, this.leverArmColor);
                }

                if (Application.isPlaying && this.displayTorqueVectors) {
                    if (muscle.jointTorques != null) {
                        this.RenderJointTorques(muscle.jointTorques, this.torqueVectorColor);
                    }
                }
            }
        }

        void OnValidate() {
            // Call singleton instance logic first.
            base.Instance();

            this.RegisterScanCallback();
        }

        void Reset() {
            this.runInEditMode = true;
            this.registeredMuscles = new List<MuscleTendonUnit>();

            this.ScanMuscles();
        }
    }
}
#endif