using UnityEngine;
using UnityEngine.UI;

public class Speed : MonoBehaviour
{
    public Rigidbody player;
    public Text speed;

    // Update is called once per frame
    void Update()
    {
        speed.text = (int)player.velocity.magnitude * 3.6f + " km/h";
    }
}