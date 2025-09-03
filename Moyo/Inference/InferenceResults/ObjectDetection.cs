namespace Moyo
{
    public partial class InferenceResult
    {
        public ObjectRect? objectRect { get; set; }
    }

    public class ObjectRect {
        public float xMin { get; set; } = 0.0f;
        public float yMin { get; set; } = 0.0f;
        public float xMax { get; set; } = 0.0f;
        public float yMax { get; set; } = 0.0f;
    }
}