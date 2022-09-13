using System;
using System.ComponentModel;
using System.Drawing;
using System.Windows.Forms;

namespace CrossNolClient
{
	public class FormConnect : Form
	{
		private IContainer components;

		private Button butConnect;

		private Label label1;

		private Label label2;

		private Button butCancel;

		public TextBox textHost;

		public TextBox textPort;

		public FormConnect()
			: this()
		{
			InitializeComponent();
		}

		private void butConnect_Click(object sender, EventArgs e)
		{
			((Form)this).set_DialogResult((DialogResult)1);
			((Control)this).Hide();
		}

		private void butCancel_Click(object sender, EventArgs e)
		{
			((Form)this).set_DialogResult((DialogResult)2);
			((Control)this).Hide();
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
			//IL_0038: Unknown result type (might be due to invalid IL or missing references)
			//IL_0042: Expected O, but got Unknown
			//IL_0052: Unknown result type (might be due to invalid IL or missing references)
			//IL_0076: Unknown result type (might be due to invalid IL or missing references)
			//IL_00d5: Unknown result type (might be due to invalid IL or missing references)
			//IL_00f9: Unknown result type (might be due to invalid IL or missing references)
			//IL_0129: Unknown result type (might be due to invalid IL or missing references)
			//IL_0150: Unknown result type (might be due to invalid IL or missing references)
			//IL_018f: Unknown result type (might be due to invalid IL or missing references)
			//IL_01b3: Unknown result type (might be due to invalid IL or missing references)
			//IL_01e6: Unknown result type (might be due to invalid IL or missing references)
			//IL_020a: Unknown result type (might be due to invalid IL or missing references)
			//IL_023d: Unknown result type (might be due to invalid IL or missing references)
			//IL_0261: Unknown result type (might be due to invalid IL or missing references)
			//IL_02b5: Unknown result type (might be due to invalid IL or missing references)
			//IL_02ce: Unknown result type (might be due to invalid IL or missing references)
			butConnect = new Button();
			label1 = new Label();
			textHost = new TextBox();
			label2 = new Label();
			textPort = new TextBox();
			butCancel = new Button();
			((Control)this).SuspendLayout();
			((Control)butConnect).set_Location(new Point(16, 65));
			((Control)butConnect).set_Name("butConnect");
			((Control)butConnect).set_Size(new Size(119, 23));
			((Control)butConnect).set_TabIndex(0);
			((Control)butConnect).set_Text("Войти в игру");
			((ButtonBase)butConnect).set_UseVisualStyleBackColor(true);
			((Control)butConnect).add_Click((EventHandler)butConnect_Click);
			((Control)label1).set_AutoSize(true);
			((Control)label1).set_Location(new Point(13, 13));
			((Control)label1).set_Name("label1");
			((Control)label1).set_Size(new Size(30, 13));
			((Control)label1).set_TabIndex(1);
			((Control)label1).set_Text("Хост");
			((Control)textHost).set_Location(new Point(16, 29));
			((Control)textHost).set_Name("textHost");
			((Control)textHost).set_Size(new Size(176, 20));
			((Control)textHost).set_TabIndex(2);
			((Control)textHost).set_Text("127.0.0.1");
			((Control)label2).set_AutoSize(true);
			((Control)label2).set_Location(new Point(225, 12));
			((Control)label2).set_Name("label2");
			((Control)label2).set_Size(new Size(32, 13));
			((Control)label2).set_TabIndex(3);
			((Control)label2).set_Text("Порт");
			((Control)textPort).set_Location(new Point(228, 29));
			((Control)textPort).set_Name("textPort");
			((Control)textPort).set_Size(new Size(66, 20));
			((Control)textPort).set_TabIndex(4);
			((Control)textPort).set_Text("7777");
			((Control)butCancel).set_Location(new Point(158, 65));
			((Control)butCancel).set_Name("butCancel");
			((Control)butCancel).set_Size(new Size(75, 23));
			((Control)butCancel).set_TabIndex(5);
			((Control)butCancel).set_Text("Отмена");
			((ButtonBase)butCancel).set_UseVisualStyleBackColor(true);
			((Control)butCancel).add_Click((EventHandler)butCancel_Click);
			((ContainerControl)this).set_AutoScaleDimensions(new SizeF(6f, 13f));
			((ContainerControl)this).set_AutoScaleMode((AutoScaleMode)1);
			((Form)this).set_ClientSize(new Size(330, 109));
			((Control)this).get_Controls().Add((Control)(object)butCancel);
			((Control)this).get_Controls().Add((Control)(object)textPort);
			((Control)this).get_Controls().Add((Control)(object)label2);
			((Control)this).get_Controls().Add((Control)(object)textHost);
			((Control)this).get_Controls().Add((Control)(object)label1);
			((Control)this).get_Controls().Add((Control)(object)butConnect);
			((Control)this).set_Name("FormConnect");
			((Form)this).set_StartPosition((FormStartPosition)1);
			((Control)this).set_Text("Параметры сервера");
			((Control)this).ResumeLayout(false);
			((Control)this).PerformLayout();
		}
	}
}
