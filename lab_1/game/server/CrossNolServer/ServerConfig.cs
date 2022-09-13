using System.Xml.Linq;

namespace CrossNolServer
{
	public class ServerConfig
	{
		public string host;

		public int port;

		public string hello;

		public ServerConfig()
		{
			XDocument val = XDocument.Load("config.xml");
			host = ((XContainer)val.get_Root()).Element(XName.op_Implicit("host")).get_Value();
			port = int.Parse(((XContainer)val.get_Root()).Element(XName.op_Implicit("port")).get_Value());
			hello = ((XContainer)val.get_Root()).Element(XName.op_Implicit("hello")).get_Value();
		}
	}
}
