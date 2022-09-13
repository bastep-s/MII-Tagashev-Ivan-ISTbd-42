using System;
using System.Net;
using System.Net.Sockets;
using System.Text;

namespace CrossNolServer
{
	internal class Program
	{
		private static int nextplayern = 1;

		private static int GAMESIZE = 9;

		private static int[] servergame = new int[GAMESIZE];

		private static string processCommand(string command)
		{
			string text = "error";
			if (command.Equals("register") && nextplayern <= 2)
			{
				text = nextplayern.ToString("D");
				nextplayern++;
			}
			if (command.Equals("getgame"))
			{
				text = "";
				for (int i = 0; i < GAMESIZE; i++)
				{
					text += servergame[i].ToString("D");
				}
			}
			if (command.StartsWith("click"))
			{
				string[] array = command.Split(new char[1]
				{
					','
				});
				int num = int.Parse(array[1]);
				int num2 = int.Parse(array[2]);
				int num3 = int.Parse(array[3]);
				servergame[num * 3 + num2] = num3;
			}
			return text;
		}

		private static void Main(string[] args)
		{
			//IL_0017: Unknown result type (might be due to invalid IL or missing references)
			//IL_001d: Expected O, but got Unknown
			//IL_0020: Unknown result type (might be due to invalid IL or missing references)
			//IL_0026: Expected O, but got Unknown
			ServerConfig serverConfig = new ServerConfig();
			IPEndPoint val = new IPEndPoint(IPAddress.Parse(serverConfig.host), serverConfig.port);
			Socket val2 = new Socket((AddressFamily)2, (SocketType)1, (ProtocolType)6);
			try
			{
				val2.Bind((EndPoint)(object)val);
				val2.Listen(10);
				for (int i = 0; i < GAMESIZE; i++)
				{
					servergame[i] = 0;
				}
				Console.WriteLine("Игра создана. Ожидаем игроков.");
				while (true)
				{
					Socket val3 = val2.Accept();
					StringBuilder stringBuilder = new StringBuilder();
					int num = 0;
					byte[] array = new byte[256];
					do
					{
						num = val3.Receive(array);
						stringBuilder.Append(Encoding.ASCII.GetString(array, 0, num));
					}
					while (val3.get_Available() > 0);
					string text = stringBuilder.ToString();
					Console.WriteLine("Получена команда: " + text);
					string s = processCommand(text);
					array = Encoding.ASCII.GetBytes(s);
					val3.Send(array);
					val3.Shutdown((SocketShutdown)2);
					val3.Close();
				}
			}
			catch (Exception ex)
			{
				Console.WriteLine(ex.Message);
			}
		}
	}
}
