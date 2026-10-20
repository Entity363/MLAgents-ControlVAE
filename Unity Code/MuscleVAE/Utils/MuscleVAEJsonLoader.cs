using UnityEngine;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Kinesis.Components;
using Kinesis.Core;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.SideChannels;
using UnityEditor;
using static MuscleVAE.Utils.MotionUtils;

public class MuscleVAEJsonLoader : MonoBehaviour
{
#if UNITY_EDITOR
    [MenuItem("MuscleVAE/Load Json")]
    static void LoadJson()
    {
        string path = EditorUtility.OpenFilePanel("Select JSON", "", "json");

        if (string.IsNullOrEmpty(path)) return;

        string json = File.ReadAllText(path);

        var root = JsonUtility.FromJson<MuscleVAEJsonData.Root>(json);
        var character = root.CharacterList.Characters[0];

        Debug.Log("Character name: " + character.CharacterName);
        
        Debug.Log("Num bodies: " + character.Bodies.Length);
        Debug.Log("Num joints: " + character.Joints.Length);
        Debug.Log("Num muscles: " + character.Muscles.Length);
        
        GameObject Character = new GameObject(character.CharacterName);

        Dictionary<int, GameObject> bodyMap = new Dictionary<int, GameObject>();
        Dictionary<string, GameObject> bodyNameMap = new Dictionary<string, GameObject>();
        
        for (int i = 0; i < character.Bodies.Length; i++)
        {
            var body = character.Bodies[i];
            GameObject go = new GameObject(body.Name);
            
            go.transform.position = ToVec3(body.Position);
            go.transform.rotation = ToQuat(body.Quaternion);
            
            Rigidbody rb = go.AddComponent<Rigidbody>();
            rb.mass = body.Mass;
            rb.inertiaTensor = Vector3.one * body.Mass;
            
            bodyMap[i] = go;
            bodyNameMap[body.Name] = go;
            
            foreach (var geom in body.Geoms)
            {
                if (!geom.Collidable) continue;

                if (geom.GeomType == "Sphere")
                {
                    var col = go.AddComponent<SphereCollider>();

                    col.center = ToVec3(geom.Position) - go.transform.position;
                    col.radius = geom.Scale[0];
                    
                    GameObject vis = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                    vis.transform.parent = col.transform;

                    vis.transform.localPosition = Vector3.zero;
                    vis.transform.localRotation = Quaternion.identity;

                    float r = geom.Scale[0];
                    vis.transform.localScale = Vector3.one * (r * 2f);
                }
                else if (geom.GeomType == "Cube")
                {
                    GameObject colGO = new GameObject(geom.Name);
                    colGO.transform.parent = go.transform;

                    // Convert world → local
                    Vector3 localPos = ToVec3(geom.Position) - ToVec3(body.Position);
                    Quaternion localRot = Quaternion.Inverse(ToQuat(body.Quaternion)) * ToQuat(geom.Quaternion);

                    colGO.transform.localPosition = localPos;
                    colGO.transform.localRotation = localRot;

                    var col = colGO.AddComponent<BoxCollider>();
                    col.size = ToVec3(geom.Scale);
                    
                    GameObject vis = GameObject.CreatePrimitive(PrimitiveType.Cube);
                    vis.transform.parent = colGO.transform;

                    vis.transform.localPosition = Vector3.zero;
                    vis.transform.localRotation = Quaternion.identity;

                    vis.transform.localScale = ToVec3(geom.Scale);
                }
                else if (geom.GeomType == "Capsule")
                {
                    GameObject colGO = new GameObject(geom.Name);
                    colGO.transform.parent = go.transform;

                    Vector3 worldPos = ToVec3(geom.Position);
                    Quaternion worldRot = ToQuat(geom.Quaternion);

                    colGO.transform.position = worldPos;
                    colGO.transform.rotation = worldRot;

                    CapsuleCollider col = colGO.AddComponent<CapsuleCollider>();

                    float radius = geom.Scale[0];
                    float height = geom.Scale[1];

                    col.radius = radius;
                    col.height = height + 2f * radius;
                    col.direction = 2;

                    // ---------------- VISUAL ----------------
                    Vector3 axis = Vector3.up;

                    if (col.direction == 0) axis = Vector3.right;
                    else if (col.direction == 1) axis = Vector3.up;
                    else if (col.direction == 2) axis = Vector3.forward;
                    
                    GameObject vis = GameObject.CreatePrimitive(PrimitiveType.Capsule);

                    vis.transform.parent = colGO.transform;

                    vis.transform.localPosition = Vector3.zero;
                    vis.transform.localRotation = Quaternion.FromToRotation(Vector3.up, axis);

                    vis.transform.localScale = new Vector3(
                        radius * 2f,
                        (height + 2f * radius) / 2f,
                        radius * 2f
                    );
                }
            }
        }
        for (int i = 0; i < character.Bodies.Length; i++)
        {
            var body = character.Bodies[i];
            GameObject go = bodyMap[i];

            if (body.ParentBodyID == -1)
            {
                go.transform.parent = Character.transform;
            }
            else
            {
                GameObject parent = bodyMap[body.ParentBodyID];
                go.transform.parent = parent.transform;
            }
        }

        foreach (var joint in character.Joints)
        {
            GameObject child = bodyMap[joint.ChildBodyID];
            GameObject parent = bodyMap[joint.ParentBodyID];

            Rigidbody parentRb = parent.GetComponent<Rigidbody>();

            ConfigurableJoint cj = child.AddComponent<ConfigurableJoint>();
            cj.connectedBody = parentRb;

            Vector3 worldAnchor = ToVec3(joint.Position);

            cj.autoConfigureConnectedAnchor = false;

            cj.anchor = child.transform.InverseTransformPoint(worldAnchor);
            cj.connectedAnchor = parent.transform.InverseTransformPoint(worldAnchor);

            cj.xMotion = ConfigurableJointMotion.Locked;
            cj.yMotion = ConfigurableJointMotion.Locked;
            cj.zMotion = ConfigurableJointMotion.Locked;

            cj.angularXMotion = ConfigurableJointMotion.Limited;
            cj.angularYMotion = ConfigurableJointMotion.Limited;
            cj.angularZMotion = ConfigurableJointMotion.Limited;

            Vector3 axis = Vector3.zero;
            if (joint.JointType == "HingeJoint")
            {            
                if (joint.EulerOrder == "X")
                {
                    axis = Vector3.right;
                }
                else if (joint.EulerOrder == "Y")
                {
                    axis = Vector3.up;
                }
                else if (joint.EulerOrder == "Z")
                {
                    axis =  Vector3.forward;
                }
                else
                {
                    Debug.LogError($"{joint.Name} has {joint.EulerOrder} axis");
                }

                axis = -axis;
                cj.axis = child.transform.InverseTransformDirection(axis);
                
                SoftJointLimit low = new SoftJointLimit();
                SoftJointLimit high = new SoftJointLimit();

                low.limit = joint.AngleLoLimit[0];
                high.limit = joint.AngleHiLimit[0];

                cj.lowAngularXLimit = low;
                cj.highAngularXLimit = high;
            }
            else if (joint.JointType == "BallJoint")
            {
                SoftJointLimit low = new SoftJointLimit();
                SoftJointLimit high = new SoftJointLimit();

                low.limit = Mathf.Max(
                    Mathf.Abs(joint.AngleLoLimit[0]),
                    Mathf.Abs(joint.AngleLoLimit[1]),
                    Mathf.Abs(joint.AngleLoLimit[2])
                );

                high.limit = Mathf.Max(
                    Mathf.Abs(joint.AngleHiLimit[0]),
                    Mathf.Abs(joint.AngleHiLimit[1]),
                    Mathf.Abs(joint.AngleHiLimit[2])
                );
                
                cj.lowAngularXLimit = new SoftJointLimit { limit = -low.limit };
                cj.highAngularXLimit = new SoftJointLimit { limit = high.limit };

                cj.angularYLimit = new SoftJointLimit { limit = high.limit };
                cj.angularZLimit = new SoftJointLimit { limit = high.limit };
                
            }

            /*
            cj.rotationDriveMode = RotationDriveMode.Slerp;
            JointDrive slerp = new JointDrive();
            slerp.positionSpring = character.PDControlParam.Kps[joint.JointID];
            slerp.positionDamper = character.PDControlParam.Kps[joint.JointID] / 10;
            slerp.maximumForce = character.PDControlParam.TorqueLimit[joint.JointID];
            cj.slerpDrive = slerp;
            */

            child.AddComponent<JointData>();

        }
        
        //Create base
        GameObject muscolature = new GameObject("Muscolature");
        muscolature.transform.parent = Character.transform;
        muscolature.AddComponent<MuscleGroup>();
        
        //Create groups
        Dictionary<string, GameObject> groupMap = new Dictionary<string, GameObject>();
        foreach (var muscle in character.Muscles)
        {
            string groupName = muscle.group_name;

            if (!groupMap.ContainsKey(groupName))
            {
                GameObject groupGO = new GameObject(groupName);
                groupGO.transform.parent = muscolature.transform;

                groupGO.AddComponent<MuscleGroup>();
                
                groupMap[groupName] = groupGO;
            }
        }
        //Add muscles in order
        foreach (var muscle in character.Muscles)
        {
            string groupName = muscle.group_name;

            GameObject groupGO = groupMap[groupName];

            GameObject muscleGO = new GameObject(muscle.Name);
            muscleGO.transform.parent = groupGO.transform;

            var muscleTendonUnit = muscleGO.AddComponent<MuscleTendonUnit>();
            muscleTendonUnit.maxIsometricForce = muscle.f0;
            if (muscle.lm > 1)
            {
                muscleTendonUnit.normalizedSEESlackLength = 0.5f;
            }
            else
            {
                muscleTendonUnit.normalizedSEESlackLength = muscle.lt;
            }
            muscleTendonUnit.l0 = muscle.lm;

            muscleTendonUnit.muscleNodes = new List<MuscleNode>();

            foreach (var waypoint in muscle.Waypoints)
            {
                GameObject bone = bodyNameMap[waypoint.BodyName]; 

                Vector3 worldPos = ToVec3(waypoint.Position);

                Vector3 localOffset = bone.transform.InverseTransformPoint(worldPos);

                MuscleNode node = new MuscleNode
                {
                    bone = bone,
                    offset = localOffset
                };

                muscleTendonUnit.muscleNodes.Add(node);
            }

            var stimulator = muscleGO.AddComponent<MuscleStimulator>();
            stimulator.muscle = muscleTendonUnit;
        }
        
        
    }
#endif
}
