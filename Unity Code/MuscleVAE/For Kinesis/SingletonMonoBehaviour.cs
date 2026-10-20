using UnityEngine;

#if UNITY_EDITOR
namespace Kinesis {
    /// <summary>
    /// Base class for providing singleton behavior to MonoBehaviour objects.
    /// </summary>
    public class SingletonMonoBehaviour<T> : MonoBehaviour where T : SingletonMonoBehaviour<T> {
        protected static T _instance;

        protected void OnDestroy() {
            _instance = null;
        }

        protected void Instance() {
            if (_instance != null && _instance != this) {
                if (Application.isPlaying) {
                    Destroy(this);
                } else {
                    // Can't call DestroyImmediate directly from OnValidate.
                    UnityEditor.EditorApplication.delayCall += () => {
                        DestroyImmediate(this);
                    };
                }

                string errorMsg = "An instance of this singleton already exists.";
                throw new System.InvalidOperationException(errorMsg);
            } else {
                _instance = (T)this;
            }
        }
    }
}
#endif