using System;
using System.Collections.Generic;
using System.ComponentModel;
using Unity.MLAgents.Actuators;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;
using Random = UnityEngine.Random;

namespace MuscleVAE.Utils
{
    public class IndexCounter
    {
        public static int[] CalculateFeasibleIndex(int[] done_flag, int rollout_length)
        {
            int n = done_flag.Length;

            int[] res_flag = new int[n];
            for (int i = 0; i < n; i++)
            {
                res_flag[i] = 1;
            }

            for (int t = 0; t < n; t++)
            {
                if (done_flag[t] == 0f) continue;

                int startMin = t - (rollout_length - 1);
                int startMax = t;

                if (startMin < 0)
                {
                    startMin = 0;
                }
                if (startMax >= n)
                {
                    startMax = n - 1;
                }

                for (int s = startMin; s <= startMax; s++)
                {
                    res_flag[s] = 0;
                }
            }

            List<int> feasible = new List<int>();
            for (int i = 0; i < n; i++)
            {
                if (res_flag[i] != 0)
                {
                    feasible.Add(i);
                }
            }

            return feasible.ToArray();
        }

        public static int RandomSelect(int[] feasible, float[] p = null)
        {
            if (p == null)
            {
                int i = UnityEngine.Random.Range(0, feasible.Length);
                return feasible[i];
            }
            
            float r = UnityEngine.Random.value;
            float cumulative = 0f;
            for (int i = 0; i < p.Length; i++)
            {
                cumulative += p[i];

                if (r <= cumulative)
                {
                    return feasible[i];
                }
            }
            
            return feasible[feasible.Length - 1];
        }
    }

    public class MotionUtils
    {

        [Serializable]
        public class BoneMapEntry
        {
            public Transform stateBone;
            public Transform targetBone;
            public float weight = 1.0f;

        }
        
        public static Vector3 ToVec3(float[] v)
        {
            return new Vector3(v[0], v[1], v[2]);
        }

        public static Quaternion ToQuat(float[] q)
        {
            return new Quaternion(q[0], q[1], q[2], q[3]);
        }

        public static Quaternion FlipQuatByW(Quaternion q)
        {
            if (q.w < 0f)
            {
                return new Quaternion(-q.x, -q.y, -q.z, -q.w);
            }

            return q;
        }

        public static Quaternion RotVecToQuat(Vector3 rotVec)
        {
            float angle = rotVec.magnitude; // radians
            if (angle < 1e-8f)
                return Quaternion.identity;

            Vector3 axis = rotVec / angle;
            return Quaternion.AngleAxis(angle * Mathf.Rad2Deg, axis);
        }

        public static Vector3 BroadcastQuatApply(Quaternion q, Vector3 v)
        {
            Vector3 qxyz = new Vector3(q.x, q.y, q.z);
            Vector3 t = 2f * Vector3.Cross(qxyz, v);
            return v + q.w * t + Vector3.Cross(qxyz, t);
        }

        public static Quaternion BroadcastQuatMultiply(Quaternion p, Quaternion q)
        {
            Vector3 pv = new Vector3(p.x, p.y, p.z);
            Vector3 qv = new Vector3(q.x, q.y, q.z);

            float w = p.w * q.w - Vector3.Dot(pv, qv);
            Vector3 xyz = p.w * qv + q.w * pv + Vector3.Cross(pv, qv);

            return new Quaternion(xyz.x, xyz.y, xyz.z, w);
        }

        public static List<float> Compute6DRotation(Quaternion quat)
        {
            Matrix4x4 M = Matrix4x4.Rotate(quat);

            List<float> rotations = new List<float>(6)
            {
                M.m00, M.m01,
                M.m10, M.m11,
                M.m20, M.m21
            };

            return rotations;
        }

        public static List<float> Compute6DRotationRow(Quaternion quat)
        {
            Matrix4x4 M = Matrix4x4.Rotate(quat);
            Vector4 M1 = M.GetRow(0);
            Vector4 M2 = M.GetRow(1);

            List<float> rotations = new List<float>();
            rotations.Add(M1.x);
            rotations.Add(M1.y);
            rotations.Add(M1.z);
            rotations.Add(M2.x);
            rotations.Add(M2.y);
            rotations.Add(M2.z);

            return rotations;
        }

        public static Vector3 GetNumericalVelocity(Vector3 current, Vector3 previous, float timeStep = 0.02f)
        {
            Vector3 vel = (current - previous) / timeStep;
            return vel;
        }

        public static Vector3 GetNumericalAngularVelocity(Quaternion current, Quaternion previous,
            float timeStep = 0.02f)
        {
            // qDelta = qCur * inverse(qPrev)
            Quaternion qd = current * Quaternion.Inverse(previous);

            // shortest path (flip by w)
            if (qd.w < 0f) qd = new Quaternion(-qd.x, -qd.y, -qd.z, -qd.w);

            // normalize just in case
            float norm = Mathf.Sqrt(qd.x * qd.x + qd.y * qd.y + qd.z * qd.z + qd.w * qd.w);
            if (norm > 1e-12f)
            {
                qd.x /= norm;
                qd.y /= norm;
                qd.z /= norm;
                qd.w /= norm;
            }

            // angle
            float w = Mathf.Clamp(qd.w, -1f, 1f);
            float angle = 2f * Mathf.Acos(w); // radians, [0, pi]

            // axis
            float s = Mathf.Sqrt(Mathf.Max(1e-12f, 1f - w * w)); // = sin(angle/2)
            Vector3 axis = (s < 1e-6f) ? new Vector3(1f, 0f, 0f) : new Vector3(qd.x, qd.y, qd.z) / s;

            return axis * (angle / timeStep); // rad/s
        }

        public static List<float> normalize(List<float> vector, List<float> mean, List<float> std, float eps = 1e-8f)
        {
            //Assert.AreEqual(vector.Count,std.Count);
            //Assert.AreEqual(vector.Count, mean.Count);
            //Assert.AreEqual(mean.Count, std.Count);

            List<float> normalize = new List<float>();
            for (int i = 0; i < vector.Count; i++)
            {
                float normalized = (vector[i] - mean[i]) / (std[i] + eps);
                normalize.Add(normalized);
            }

            return normalize;
        }

        public static int GetAvg(List<int> i)
        {
            if (i.Count < 1)
            {
                return 2048; //use an impossible one to know if it doesn't work
            }

            int avg = 0;
            for (int j = 0; j < i.Count; j++)
            {
                avg += i[j];
            }

            avg /= i.Count;
            return avg;
        }

        public static float MeanAbsError(List<float> a, List<float> b)
        {
            Debug.Assert(a.Count == b.Count);

            float sum = 0f;
            for (int i = 0; i < a.Count; i++)
            {
                sum += Mathf.Abs(a[i] - b[i]);
            }

            return sum / a.Count;
        }


        public static List<float> Straighten2List(List<List<float>> buffer)
        {
            List<float> result = new List<float>();
            foreach (var body in buffer)
            {
                foreach (var obs in body)
                {
                    result.Add(obs);
                }
            }

            return result;
        }

        public static List<List<float>> ComputeObsFromState(List<float> state)
        {
            //Debug.LogError("Fix Obs to match state2ob 16 x n + 3");
            Vector3 rootPos = new Vector3(state[0], state[1], state[2]);
            Quaternion rootRot = new Quaternion(state[3], state[4], state[5], state[6]);
            Quaternion root_rot_inv = Quaternion.Inverse(rootRot);

            List<List<float>> obs = new List<List<float>>();
            int count = state.Count / 13;
            int k = 0;
            for (int i = 0; i < count; i++)
            {
                List<float> buffer = new List<float>();
                Vector3 pos = new Vector3(state[k], state[k + 1], state[k + 2]);
                Quaternion rot = new Quaternion(state[k + 3], state[k + 4], state[k + 5], state[k + 6]);
                Vector3 vel = new Vector3(state[k + 7], state[k + 8], state[k + 9]); //_baseBoneVels[i];
                Vector3 avel = new Vector3(state[k + 10], state[k + 11], state[k + 12]); //_baseBoneAvels[i];

                Vector3 local_pos = BroadcastQuatApply(root_rot_inv, pos - rootPos);
                Vector3 local_vel = BroadcastQuatApply(root_rot_inv, vel);
                Vector3 local_avel = BroadcastQuatApply(root_rot_inv, avel);
                Quaternion local_rot = FlipQuatByW(BroadcastQuatMultiply(root_rot_inv, rot));
                var local_rot6D = Compute6DRotation(local_rot);
                float height = pos.y;

                //3 pos + 6 rot + 3 vel + 3 avel + 1 height = 16
                buffer.Add(local_pos.x);
                buffer.Add(local_pos.y);
                buffer.Add(local_pos.z);
                buffer.Add(local_rot6D[0]);
                buffer.Add(local_rot6D[1]);
                buffer.Add(local_rot6D[2]);
                buffer.Add(local_rot6D[3]);
                buffer.Add(local_rot6D[4]);
                buffer.Add(local_rot6D[5]);
                buffer.Add(local_vel.x);
                buffer.Add(local_vel.y);
                buffer.Add(local_vel.z);
                buffer.Add(local_avel.x);
                buffer.Add(local_avel.y);
                buffer.Add(local_avel.z);
                buffer.Add(height);

                obs.Add(buffer);

                k += 13;
            }

            Vector3 updDir = new Vector3(0, 1, 0);
            Vector3 localUpDir = BroadcastQuatApply(root_rot_inv, updDir);
            List<float> localUpDirs = new List<float>();
            localUpDirs.Add(localUpDir.x);
            localUpDirs.Add(localUpDir.y);
            localUpDirs.Add(localUpDir.z);

            obs.Add(localUpDirs);


            return obs;
        }

        // python state2ob is all pos, all rot, all vel, all avel, all height, updir
        public static List<List<float>> state2ob(List<float> state)
        {
            Vector3 rootPos = new Vector3(state[0], state[1], state[2]);
            Quaternion rootRot = new Quaternion(state[3], state[4], state[5], state[6]);
            Quaternion root_rot_inv = Quaternion.Inverse(rootRot);

            int count = state.Count / 13;
            int k = 0;

            List<float> posBuffer = new List<float>(count * 3);
            List<float> rotBuffer = new List<float>(count * 6);
            List<float> velBuffer = new List<float>(count * 3);
            List<float> avelBuffer = new List<float>(count * 3);
            List<float> heightBuffer = new List<float>(count);

            for (int i = 0; i < count; i++)
            {
                Vector3 pos = new Vector3(state[k], state[k + 1], state[k + 2]);
                Quaternion rot = new Quaternion(state[k + 3], state[k + 4], state[k + 5], state[k + 6]);
                Vector3 vel = new Vector3(state[k + 7], state[k + 8], state[k + 9]);
                Vector3 avel = new Vector3(state[k + 10], state[k + 11], state[k + 12]);

                Vector3 local_pos = BroadcastQuatApply(root_rot_inv, pos - rootPos);
                Vector3 local_vel = BroadcastQuatApply(root_rot_inv, vel);
                Vector3 local_avel = BroadcastQuatApply(root_rot_inv, avel);
                Quaternion local_rot = FlipQuatByW(BroadcastQuatMultiply(root_rot_inv, rot));
                var local_rot6D = Compute6DRotation(local_rot);
                float height = pos.y;

                posBuffer.Add(local_pos.x);
                posBuffer.Add(local_pos.y);
                posBuffer.Add(local_pos.z);

                rotBuffer.Add(local_rot6D[0]);
                rotBuffer.Add(local_rot6D[1]);
                rotBuffer.Add(local_rot6D[2]);
                rotBuffer.Add(local_rot6D[3]);
                rotBuffer.Add(local_rot6D[4]);
                rotBuffer.Add(local_rot6D[5]);

                velBuffer.Add(local_vel.x);
                velBuffer.Add(local_vel.y);
                velBuffer.Add(local_vel.z);

                avelBuffer.Add(local_avel.x);
                avelBuffer.Add(local_avel.y);
                avelBuffer.Add(local_avel.z);

                heightBuffer.Add(height);

                k += 13;
            }

            Vector3 updDir = new Vector3(0, 1, 0);
            Vector3 localUpDir = BroadcastQuatApply(root_rot_inv, updDir);
            List<float> upBuffer = new List<float>(3)
            {
                localUpDir.x, localUpDir.y, localUpDir.z
            };

            return new List<List<float>>
            {
                posBuffer,
                rotBuffer,
                velBuffer,
                avelBuffer,
                heightBuffer,
                upBuffer
            };
        }


        public static List<List<float>> decompose_obs(List<float> obs)
        {
            Debug.Assert((obs.Count - 3) % 16 == 0);
            int num_body = (obs.Count - 3) / 16;

            int posSz = num_body * 3;
            int rotSz = num_body * 6;
            int velSz = num_body * 3;
            int avelSz = num_body * 3;
            int heightSz = num_body * 1;
            int upSz = 3;

            int offset = 0;

            var pos = obs.GetRange(offset, posSz);
            offset += posSz;
            var rots = obs.GetRange(offset, rotSz);
            offset += rotSz;
            var vels = obs.GetRange(offset, velSz);
            offset += velSz;
            var avels = obs.GetRange(offset, avelSz);
            offset += avelSz;
            var heights = obs.GetRange(offset, heightSz);
            offset += heightSz;
            var ups = obs.GetRange(offset, upSz);

            return new List<List<float>> { pos, rots, vels, avels, heights, ups };
        }

        public static float pose_err(List<float> state, List<float> target, float posW = 0.2f, float rotW = 0.1f,
            float velW = 0.5f, float avelW = 0.5f, float heightW = 1.2f, float upDirW = 3f, float dt = 1f / 50f)
        {
            List<List<float>> baseBuffer = decompose_obs(state);
            List<List<float>> targetBuffer = decompose_obs(target);

            float posErr = MeanAbsError(baseBuffer[0], targetBuffer[0]);
            float rotErr = MeanAbsError(baseBuffer[1], targetBuffer[1]);
            float velErr = MeanAbsError(baseBuffer[2], targetBuffer[2]);
            float avelErr = MeanAbsError(baseBuffer[3], targetBuffer[3]);
            float heightErr = MeanAbsError(baseBuffer[4], targetBuffer[4]);
            float upErr = MeanAbsError(baseBuffer[5], targetBuffer[5]);

            float totalErr =
                posErr * posW +
                rotErr * rotW +
                velErr * velW +
                avelErr * avelW +
                heightErr * heightW +
                upErr * upDirW;

            return totalErr;
        }


        public static float PoseError(List<List<float>> stateObs, List<List<float>> targetObs, float posW = 0.2f,
            float rotW = 0.1f, float velW = 0.5f, float avelW = 0.5f, float heightW = 1.2f, float upDirW = 3f,
            float dt = 1f / 20f)
        {
            int Count = Mathf.Min(stateObs.Count, targetObs.Count);
            float posLoss = 0;
            float rotLoss = 0;
            float velLoss = 0;
            float avelLoss = 0;
            float heightLoss = 0;
            float upDirLoss = 0;

            for (int j = 0; j < Count - 1; j++)
            {
                for (int i = 0; i < 3; i++) //pos
                {
                    float delta_pos = stateObs[j][i] - targetObs[j][i];
                    float p = Mathf.Abs(delta_pos);
                    posLoss += p;
                }

                for (int i = 3; i < 9; i++) //rot
                {
                    float delta_rot = stateObs[j][i] - targetObs[j][i];
                    float r = Mathf.Abs(delta_rot);
                    rotLoss += r;
                }

                for (int i = 9; i < 12; i++) //vel
                {
                    float delta_vel = stateObs[j][i] - targetObs[j][i];
                    float v = Mathf.Abs(delta_vel);
                    velLoss += v;
                }

                for (int i = 12; i < 15; i++) //avel
                {
                    float delta_avel = stateObs[j][i] - targetObs[j][i];
                    float a = Mathf.Abs(delta_avel);
                    avelLoss += a;
                }

                //height
                float delta_h = stateObs[j][15] - targetObs[j][15];
                float h = Mathf.Abs(delta_h);
                heightLoss += h;
            }

            for (int i = 0; i < 3; i++)
            {
                float delta_updir = stateObs[Count - 1][i] - targetObs[Count - 1][i];
                float u = Mathf.Abs(delta_updir);
                upDirLoss += u;
            }

            posLoss /= 3f * Count;
            rotLoss /= 6f * Count;
            velLoss /= 3f * Count;
            avelLoss /= 3f * Count;
            heightLoss /= 1f * Count;
            upDirLoss /= 3f;

            float rwd = posW * posLoss + rotW * rotLoss + velW * velLoss * dt + avelW * avelLoss * dt +
                        heightW * heightLoss + upDirW * upDirLoss;
            return rwd;
        }


        public static List<float> RandomTarget(List<float> speedRange)
        {
            float speed = Random.Range(0, speedRange.Count);
            float direction = Random.Range(0, Mathf.PI * 2);
            return new List<float> { speed, direction };
        }

    }
}