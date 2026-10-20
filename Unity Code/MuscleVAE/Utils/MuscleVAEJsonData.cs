using UnityEngine;

public class MuscleVAEJsonData
{
    [System.Serializable]
    public class Root
    {
        public CharacterList CharacterList;
    }
    
    [System.Serializable]
    public class CharacterList
    {
        public Character[] Characters;
    }
    
    [System.Serializable]
    public class Character
    {
        public int CharacterID;
        public string CharacterName;
        public BodyData[] Bodies;
        public JointData[] Joints;
        public MuscleData[] Muscles;
        public PDControlParamData PDControlParam;
    }
    
    [System.Serializable]
    public class BodyData
    {
        public string Name;
        public float Mass;
        public Inertia Inertia;
        public int ParentJointID;
        public int ParentBodyID;
        public float[] Position;
        public float[] Quaternion;
        public GeometryData[] Geoms;
    }

    [System.Serializable]
    public class Inertia
    {
        public float[] matrix =
        {
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        };
    }
    
    [System.Serializable]
    public class GeometryData
    {
        public int GeomID;
        public string Name;
        public string GeomType;
        public bool Collidable;
        public float Friction;
        public float Restitution;
        public float[] Position;
        public float[] Quaternion;
        public float[] Scale;
    }

    [System.Serializable]
    public class JointData
    {
        public int JointID;
        public string Name;
        public string JointType;
        public float Damping;
        public float Weight;
        public float[] Position;
        public float[] Quaternion;
        public float[] AngleLoLimit;
        public float[] AngleHiLimit;
        public string EulerOrder;
        public float[] EulerAxisLocalRot;
        public int ParentBodyID;
        public int ChildBodyID;
        public int ParentJointID;
    }

    [System.Serializable]
    public class MuscleData
    {
        public int muscleID;
        public string Name;
        public string group_name;
        public float f0;
        public float lm;
        public float lt;
        public float lmax;
        public WayPointData[] Waypoints;
    }

    [System.Serializable]
    public class WayPointData
    {
        public int waypointID;
        public string BodyName;
        public float[] Position;
    }

    [System.Serializable]
    public class PDControlParamData
    {
        public float[] Kps;
        public float[] TorqueLimit;
    }
    
}
