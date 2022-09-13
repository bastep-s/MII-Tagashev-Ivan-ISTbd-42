using System;
using System.ComponentModel;
using System.Drawing;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using System.Windows.Forms;

namespace CrossNolClient
{
	public class FormMain : Form
	{
		private IPEndPoint serverpoint;

		private GamePainter gp;

		private Game game;

		private bool mystep;

		private bool gameover;

		private IContainer components;

		private Panel panel1;

		private Panel panelPaint;

		private Label labInfo;

		private Button butConnect;

		private Timer timer1;

		public FormMain()
			: this()
		{
			InitializeComponent();
		}

		private void SetDoubleBuffered(Control c, bool value)
		{
			PropertyInfo property = typeof(Control).GetProperty("DoubleBuffered", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.SetProperty);
			if (property != null)
			{
				property.SetValue(c, value, null);
			}
		}

		private void updateInfo()
		{
			((Control)labInfo).set_Text("Ваша фигура: " + game.getPlayerFigure() + Environment.NewLine);
			bool flag = false;
			if (game.isGameOver(flag))
			{
				gameover = true;
				Label obj = labInfo;
				((Control)obj).set_Text(((Control)obj).get_Text() + (flag ? "Вы победили" : "Вы проиграли"));
			}
			else
			{
				Label obj2 = labInfo;
				((Control)obj2).set_Text(((Control)obj2).get_Text() + (mystep ? "Сейчас ваш ход" : "Ожидаем хода соперника"));
			}
		}

		private void goNextStep()
		{
			((Control)panelPaint).Refresh();
			mystep = !mystep;
			updateInfo();
		}

		private string sendCommand(string command)
		{
			//IL_0003: Unknown result type (might be due to invalid IL or missing references)
			//IL_0009: Expected O, but got Unknown
			try
			{
				Socket val = new Socket((AddressFamily)2, (SocketType)1, (ProtocolType)6);
				val.Connect((EndPoint)(object)serverpoint);
				byte[] bytes = Encoding.ASCII.GetBytes(command);
				val.Send(bytes);
				bytes = new byte[256];
				StringBuilder stringBuilder = new StringBuilder();
				int num = 0;
				do
				{
					num = val.Receive(bytes, bytes.Length, (SocketFlags)0);
					stringBuilder.Append(Encoding.ASCII.GetString(bytes, 0, num));
				}
				while (val.get_Available() > 0);
				val.Shutdown((SocketShutdown)2);
				val.Close();
				return stringBuilder.ToString();
			}
			catch (Exception)
			{
				return "error";
			}
		}

		private void panelPaint_Paint(object sender, PaintEventArgs e)
		{
			if (gp != null)
			{
				gp.RenderTo(e.get_Graphics(), ((Control)panelPaint).get_Width(), ((Control)panelPaint).get_Height());
			}
		}

		private void FormMain_Load(object sender, EventArgs e)
		{
			SetDoubleBuffered((Control)(object)panelPaint, value: true);
		}

		private void panelPaint_MouseClick(object sender, MouseEventArgs e)
		{
		}

		private void panelPaint_MouseUp(object sender, MouseEventArgs e)
		{
			if (!mystep || gameover)
			{
				return;
			}
			int celli = 0;
			int cellj = 0;
			if (gp.getClickIJ(e.get_X(), e.get_Y(), ((Control)panelPaint).get_Width(), ((Control)panelPaint).get_Height(), ref celli, ref cellj))
			{
				string command = "";
				if (game.doClick(celli, cellj, ref command))
				{
					sendCommand(command);
				}
			}
		}

		private void panelPaint_Resize(object sender, EventArgs e)
		{
			((Control)panelPaint).Refresh();
		}

		private void butConnect_Click(object sender, EventArgs e)
		{
			//IL_0007: Unknown result type (might be due to invalid IL or missing references)
			//IL_000d: Invalid comparison between Unknown and I4
			//IL_003d: Unknown result type (might be due to invalid IL or missing references)
			//IL_0047: Expected O, but got Unknown
			//IL_0065: Unknown result type (might be due to invalid IL or missing references)
			FormConnect formConnect = new FormConnect();
			if ((int)((Form)formConnect).ShowDialog() == 1)
			{
				serverpoint = new IPEndPoint(IPAddress.Parse(((Control)formConnect.textHost).get_Text().Trim()), int.Parse(((Control)formConnect.textPort).get_Text().Trim()));
				string text = sendCommand("register");
				if (text.Equals("error"))
				{
					MessageBox.Show("Ошибка связи с сервером");
					return;
				}
				game = new Game(int.Parse(text), goNextStep);
				mystep = game.getPlayerN() == 1;
				gp = new GamePainter(game);
				((Control)panelPaint).Refresh();
				updateInfo();
				timer1.set_Enabled(true);
				((Control)butConnect).Hide();
			}
		}

		private void timer1_Tick(object sender, EventArgs e)
		{
			if (!mystep && !gameover)
			{
				timer1.set_Enabled(false);
				string gameData = sendCommand("getgame");
				game.setGameData(gameData);
				timer1.set_Enabled(true);
			}
		}

		protected override void Dispose(bool disposing)
		{
			if (disposing && components != null)
			{
				((IDisposable)components).Dispose();
			}
			((Form)this).Dispose(disposing);
		}

		private void InitializeComponent()
		{
			//IL_0001: Unknown result type (might be due to invalid IL or missing references)
			//IL_000b: Expected O, but got Unknown
			//IL_000c: Unknown result type (might be due to invalid IL or missing references)
			//IL_0016: Expected O, but got Unknown
			//IL_0017: Unknown result type (might be due to invalid IL or missing references)
			//IL_0021: Expected O, but got Unknown
			//IL_0022: Unknown result type (might be due to invalid IL or missing references)
			//IL_002c: Expected O, but got Unknown
			//IL_002d: Unknown result type (might be due to invalid IL or missing references)
			//IL_0037: Expected O, but got Unknown
			//IL_003e: Unknown result type (might be due to invalid IL or missing references)
			//IL_0048: Expected O, but got Unknown
			//IL_0099: Unknown result type (might be due to invalid IL or missing references)
			//IL_00c0: Unknown result type (might be due to invalid IL or missing references)
			//IL_00eb: Unknown result type (might be due to invalid IL or missing references)
			//IL_0115: Unknown result type (might be due to invalid IL or missing references)
			//IL_0138: Unknown result type (might be due to invalid IL or missing references)
			//IL_0142: Expected O, but got Unknown
			//IL_014f: Unknown result type (might be due to invalid IL or missing references)
			//IL_0159: Expected O, but got Unknown
			//IL_0166: Unknown result type (might be due to invalid IL or missing references)
			//IL_0170: Expected O, but got Unknown
			//IL_01aa: Unknown result type (might be due to invalid IL or missing references)
			//IL_01b4: Expected O, but got Unknown
			//IL_01be: Unknown result type (might be due to invalid IL or missing references)
			//IL_01e5: Unknown result type (might be due to invalid IL or missing references)
			//IL_0218: Unknown result type (might be due to invalid IL or missing references)
			//IL_023f: Unknown result type (might be due to invalid IL or missing references)
			//IL_02ba: Unknown result type (might be due to invalid IL or missing references)
			//IL_02d6: Unknown result type (might be due to invalid IL or missing references)
			components = (IContainer)new Container();
			panel1 = new Panel();
			panelPaint = new Panel();
			labInfo = new Label();
			butConnect = new Button();
			timer1 = new Timer(components);
			((Control)panel1).SuspendLayout();
			((Control)this).SuspendLayout();
			((Control)panel1).get_Controls().Add((Control)(object)butConnect);
			((Control)panel1).get_Controls().Add((Control)(object)labInfo);
			((Control)panel1).set_Dock((DockStyle)1);
			((Control)panel1).set_Location(new Point(0, 0));
			((Control)panel1).set_Name("panel1");
			((Control)panel1).set_Size(new Size(530, 54));
			((Control)panel1).set_TabIndex(0);
			((Control)panelPaint).set_Dock((DockStyle)5);
			((Control)panelPaint).set_Location(new Point(0, 54));
			((Control)panelPaint).set_Name("panelPaint");
			((Control)panelPaint).set_Size(new Size(530, 471));
			((Control)panelPaint).set_TabIndex(1);
			((Control)panelPaint).add_Paint(new PaintEventHandler(panelPaint_Paint));
			((Control)panelPaint).add_MouseClick(new MouseEventHandler(panelPaint_MouseClick));
			((Control)panelPaint).add_MouseUp(new MouseEventHandler(panelPaint_MouseUp));
			((Control)panelPaint).add_Resize((EventHandler)panelPaint_Resize);
			((Control)labInfo).set_AutoSize(true);
			((Control)labInfo).set_Font(new Font("Tahoma", 12f, (FontStyle)1, (GraphicsUnit)3, (byte)204));
			((Control)labInfo).set_Location(new Point(12, 9));
			((Control)labInfo).set_Name("labInfo");
			((Control)labInfo).set_Size(new Size(300, 19));
			((Control)labInfo).set_TabIndex(0);
			((Control)labInfo).set_Text("Подключитесь, чтобы начать игру");
			((Control)butConnect).set_Location(new Point(361, 12));
			((Control)butConnect).set_Name("butConnect");
			((Control)butConnect).set_Size(new Size(157, 23));
			((Control)butConnect).set_TabIndex(1);
			((Control)butConnect).set_Text("Присоединиться к серверу");
			((ButtonBase)butConnect).set_UseVisualStyleBackColor(true);
			((Control)butConnect).add_Click((EventHandler)butConnect_Click);
			timer1.set_Interval(1000);
			timer1.add_Tick((EventHandler)timer1_Tick);
			((ContainerControl)this).set_AutoScaleDimensions(new SizeF(6f, 13f));
			((ContainerControl)this).set_AutoScaleMode((AutoScaleMode)1);
			((Form)this).set_ClientSize(new Size(530, 525));
			((Control)this).get_Controls().Add((Control)(object)panelPaint);
			((Control)this).get_Controls().Add((Control)(object)panel1);
			((Control)this).set_DoubleBuffered(true);
			((Control)this).set_Name("FormMain");
			((Form)this).set_StartPosition((FormStartPosition)1);
			((Control)this).set_Text("Крестики-нолики клиент");
			((Form)this).add_Load((EventHandler)FormMain_Load);
			((Control)panel1).ResumeLayout(false);
			((Control)panel1).PerformLayout();
			((Control)this).ResumeLayout(false);
		}
	}
}
