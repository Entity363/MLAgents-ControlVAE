using UnityEngine;
using UnityEngine.InputSystem;

public class ControlVAECamera : MonoBehaviour
{
    public Transform target;

    public float distance = 4f;
    public float minDistance = 1.5f;
    public float maxDistance = 10f;

    public float sensitivity = 0.015f;
    public float zoomSpeed = 0.5f;

    public float pitchMin = -30f;
    public float pitchMax = 80f;

    [Header("Heading output for agent")]
    public float moveSpeed;
    public float moveCos;
    public float moveSin;

    [Header("speed")] 
    public float walkSpeed = 1.4f;
    public float runSpeed = 3f;

    [Header("Debug")]
    public Vector3 projectedMoveDirection;
    public Vector3 pelvisFacingDirection;

    private float yaw;
    private float pitch = 20f;

    void LateUpdate()
    {
        if (target == null || Mouse.current == null || Keyboard.current == null)
            return;

        bool rightHeld = Mouse.current.rightButton.isPressed;

        if (rightHeld)
        {
            Vector2 mouseDelta = Mouse.current.delta.ReadValue();

            yaw += mouseDelta.x * sensitivity;
            pitch -= mouseDelta.y * sensitivity;
            pitch = Mathf.Clamp(pitch, pitchMin, pitchMax);

            float scroll = Mouse.current.scroll.ReadValue().y;
            distance -= scroll * zoomSpeed;
            distance = Mathf.Clamp(distance, minDistance, maxDistance);
        }

        Quaternion rot = Quaternion.Euler(pitch, yaw, 0f);
        Vector3 offset = rot * new Vector3(0f, 0f, -distance);

        transform.position = target.position + offset;
        transform.LookAt(target.position);

        Vector3 camForward = transform.forward;
        camForward.y = 0f;
        if (camForward.sqrMagnitude > 1e-8f) camForward.Normalize();

        Vector3 camRight = transform.right;
        camRight.y = 0f;
        if (camRight.sqrMagnitude > 1e-8f) camRight.Normalize();

        float h = 0f;
        float v = 0f;

        if (Keyboard.current.aKey.isPressed) h -= 1f;
        if (Keyboard.current.dKey.isPressed) h += 1f;
        if (Keyboard.current.wKey.isPressed) v += 1f;
        if (Keyboard.current.sKey.isPressed) v -= 1f;

        Vector3 move = camForward * v + camRight * h;
        float inputMagnitude = Mathf.Clamp01(move.magnitude);

        float baseSpeed = Keyboard.current.leftShiftKey.isPressed ? runSpeed : walkSpeed;
        moveSpeed = inputMagnitude * baseSpeed;

        if (moveSpeed > 0.0001f)
        {
            projectedMoveDirection = move.normalized;

            Vector3 facing = target.forward;
            facing.y = 0f;

            if (facing.sqrMagnitude > 1e-8f)
            {
                facing.Normalize();
                pelvisFacingDirection = facing;

                moveCos = Vector3.Dot(projectedMoveDirection, facing);

                moveSin = projectedMoveDirection.x * facing.z
                        - projectedMoveDirection.z * facing.x;
            }
            else
            {
                pelvisFacingDirection = Vector3.zero;
                moveCos = 1f;
                moveSin = 0f;
            }
        }
        else
        {
            projectedMoveDirection = Vector3.zero;
            pelvisFacingDirection = Vector3.zero;

            moveCos = 1f;
            moveSin = 0f;
            moveSpeed = 0f;
        }
    }
}