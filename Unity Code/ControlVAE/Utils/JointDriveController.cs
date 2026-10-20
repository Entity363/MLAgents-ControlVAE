using UnityEngine;

public static class JointDriveController
{
    /// <summary>
    /// Sets a joint's targetRotation to match a given local rotation.
    /// The joint transform's local rotation must be cached on Start and passed into this method.
    /// </summary>
    public static void SetTargetRotationLocal(
        this ConfigurableJoint joint,
        Quaternion targetLocalRotation,
        Quaternion startLocalRotation)
    {
        if (joint.configuredInWorldSpace)
        {
            Debug.LogError(
                "SetTargetRotationLocal should not be used with joints that are configured in world space. " +
                "For world space joints, use SetTargetRotation.", joint);
        }

        SetTargetRotationInternal(joint, targetLocalRotation, startLocalRotation, Space.Self);
    }

    /// <summary>
    /// Sets a joint's targetRotation to match a given world rotation.
    /// The joint transform's world rotation must be cached on Start and passed into this method.
    /// </summary>
    public static void SetTargetRotationLocalDrive(
    this ConfigurableJoint joint,
    Quaternion targetLocalRotation,
    Quaternion startLocalRotation)
    {
        if (joint.configuredInWorldSpace)
        {
            Debug.LogError(
                "SetTargetRotationLocalDrive should not be used with joints that are configured in world space. " +
                "For world space joints, use SetTargetRotationDrive.", joint);
        }

        SetTargetRotationInternal(joint, targetLocalRotation, startLocalRotation, Space.Self);
    }


    private static void SetTargetRotationInternal(
        ConfigurableJoint joint,
        Quaternion targetRotation,
        Quaternion startRotation,
        Space space)
    {
        // ---------- SAFE AXIS BUILD (no zero vectors, no parallel vectors) ----------

        // Base right axis
        Vector3 right = joint.axis;
        if (right.sqrMagnitude < 1e-6f)
        {
            // Fallback if axis is zero/invalid
            right = Vector3.right;
        }
        else
        {
            right.Normalize();
        }

        // Secondary axis
        Vector3 secondary = joint.secondaryAxis;
        if (secondary.sqrMagnitude < 1e-6f)
        {
            secondary = Vector3.up;
        }
        else
        {
            secondary.Normalize();
        }

        // If almost parallel to 'right', pick a different secondary
        float dot = Mathf.Abs(Vector3.Dot(right, secondary));
        if (dot > 0.999f)
        {
            // choose an arbitrary axis not parallel to 'right'
            secondary = (Mathf.Abs(right.y) < 0.99f) ? Vector3.up : Vector3.forward;
        }

        // Forward = secondary � right
        Vector3 forward = Vector3.Cross(secondary, right);
        if (forward.sqrMagnitude < 1e-6f)
        {
            // Last-resort fallback
            forward = Vector3.forward;
        }
        else
        {
            forward.Normalize();
        }

        // Up = forward � right
        Vector3 up = Vector3.Cross(forward, right);
        if (up.sqrMagnitude < 1e-6f)
        {
            up = Vector3.up;
        }
        else
        {
            up.Normalize();
        }

        // Now forward & up are guaranteed non-zero & non-parallel
        Quaternion worldToJointSpace = Quaternion.LookRotation(forward, up);

        // ---------- ORIGINAL MATH (but with safe basis) ----------

        Quaternion resultRotation = Quaternion.Inverse(worldToJointSpace);

        if (space == Space.World)
        {
            resultRotation *= startRotation * Quaternion.Inverse(targetRotation);
        }
        else
        {
            resultRotation *= Quaternion.Inverse(targetRotation) * startRotation;
        }

        resultRotation *= worldToJointSpace;

        joint.targetRotation = resultRotation;
    }

    /// <summary>
    /// Adjust ConfigurableJoint settings to closely match CharacterJoint behaviour
    /// </summary>
    public static void SetupAsCharacterJoint(this ConfigurableJoint joint)
    {
        joint.xMotion = ConfigurableJointMotion.Locked;
        joint.yMotion = ConfigurableJointMotion.Locked;
        joint.zMotion = ConfigurableJointMotion.Locked;
        joint.angularXMotion = ConfigurableJointMotion.Limited;
        joint.angularYMotion = ConfigurableJointMotion.Limited;
        joint.angularZMotion = ConfigurableJointMotion.Limited;
        joint.breakForce = Mathf.Infinity;
        joint.breakTorque = Mathf.Infinity;

        joint.rotationDriveMode = RotationDriveMode.Slerp;
        var slerpDrive = joint.slerpDrive;
        slerpDrive.maximumForce = Mathf.Infinity;
        joint.slerpDrive = slerpDrive;
    }
}
