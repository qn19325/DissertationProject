using UnityEngine;
using UnityEngine.UI;

public class Speed : MonoBehaviour
{
    public Rigidbody player;
    public Text speed;

    // Update is called once per frame
    void Update()
    {
        int currentSpeed = (int)player.velocity.magnitude;
        if (currentSpeed < 100)
        {
            speed.text = "Speed: " + currentSpeed;
        } else
        {
            speed.text = "Speed: " + currentSpeed + " (Max)";
        }
    }
}