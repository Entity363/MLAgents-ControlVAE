using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using Kinesis.Components;
using Kinesis.Core;

public class MuscleGLRenderer : MonoBehaviour
{
    [Header("muscles")] public MuscleGroup group;
    
    public List<Kinesis.Core.MuscleTendonUnit> muscles;

    private Material lineMat;
    
    [Header("Colors")]
    public Color activeColor = Color.red;
    public Color deactiveColor = Color.white;

    void Awake()
    {
        lineMat = new Material(Shader.Find("Hidden/Internal-Colored"));

        lineMat.hideFlags = HideFlags.HideAndDontSave;

        lineMat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        lineMat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        lineMat.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
        lineMat.SetInt("_ZWrite", 1);
        lineMat.SetInt("_ZTest", (int)UnityEngine.Rendering.CompareFunction.LessEqual);
    }

    void OnRenderObject()
    {
        if (muscles == null) return;

        lineMat.SetPass(0);

        GL.PushMatrix();
        GL.Begin(GL.LINES);

        foreach (var muscle in muscles)
        {
            if (muscle == null || muscle.muscleNodes == null)
                continue;

            Color c = Color.Lerp(
                deactiveColor,
                activeColor,
                muscle.activation
            );

            GL.Color(c);

            for (int i = 1; i < muscle.muscleNodes.Count; i++)
            {
                var a = muscle.muscleNodes[i - 1];
                var b = muscle.muscleNodes[i];

                if (a.bone == null || b.bone == null)
                    continue;

                Vector3 p0 = a.bone.transform.TransformPoint(a.offset);
                Vector3 p1 = b.bone.transform.TransformPoint(b.offset);

                GL.Vertex(p0);
                GL.Vertex(p1);
            }
        }

        GL.End();
        GL.PopMatrix();
    }

    [ContextMenu("Fill Muscles")]
    public void FillMuscles()
    {
        //muscles = FindObjectsByType<MuscleTendonUnit>(FindObjectsSortMode.None).ToList();
        muscles = group.GetComponentsInChildren<MuscleTendonUnit>().ToList();
    }
}