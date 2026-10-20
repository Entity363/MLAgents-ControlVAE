using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using System.Text;
using System;
using Random = UnityEngine.Random;

public class FloatArraySideChannel : SideChannel
{
    public FloatArraySideChannel()
    {
        ChannelId = new Guid("621f0a70-4f87-11ea-a6bf-784f4387d1f7");
    }
    
    protected override void OnMessageReceived(IncomingMessage msg)
    {
        var receivedString = msg.ReadString();
        Debug.Log("From Python : " + receivedString);
    }

    public void SendToPython(float[] message, string key = "")
    {
        if (message == null || message.Length == 0) return;
        using (var msgOut = new OutgoingMessage())
        {
            msgOut.WriteString(key);
            msgOut.WriteFloatList(message);
            QueueMessageToSend(msgOut);
        }

        int id = Random.Range(0, message.Length);
        Debug.Log("Id: " + id + "Sent: " + message[id]);
    }
}
