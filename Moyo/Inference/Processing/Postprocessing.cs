namespace Moyo
{
    public partial class YOLO
    {
        public partial class Postprocessing
        {
            private readonly YOLO model;

            public Postprocessing(YOLO yolo)
            {
                this.model = yolo;
            }
        }
    }
}