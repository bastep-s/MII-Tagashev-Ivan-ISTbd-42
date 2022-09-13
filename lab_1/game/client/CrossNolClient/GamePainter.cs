using System;
using System.Drawing;

namespace CrossNolClient
{
	public class GamePainter
	{
		private Game game;

		private static int SPACE = 10;

		private static Pen pen = new Pen((Brush)new SolidBrush(Color.get_Black()), 2f);

		private static Pen pengray = new Pen((Brush)new SolidBrush(Color.get_Gray()), 1f);

		private static Image spr_cross = Image.FromFile("cross.png");

		private static Image spr_nol = Image.FromFile("nol.png");

		public GamePainter(Game game)
		{
			this.game = game;
		}

		public void RenderTo(Graphics g, int w, int h)
		{
			//IL_00fc: Unknown result type (might be due to invalid IL or missing references)
			//IL_011a: Unknown result type (might be due to invalid IL or missing references)
			int num = Math.Min(w, h);
			int num2 = (num - 2 * SPACE) / Game.SIZE;
			g.DrawRectangle(pengray, SPACE, SPACE, num - 2 * SPACE, num - 2 * SPACE);
			for (int i = 1; i < Game.SIZE; i++)
			{
				g.DrawLine(pen, SPACE, SPACE + i * num2, num - SPACE, SPACE + i * num2);
				g.DrawLine(pen, SPACE + i * num2, SPACE, SPACE + i * num2, num - SPACE);
			}
			Rectangle val = default(Rectangle);
			for (int j = 0; j < Game.SIZE; j++)
			{
				for (int k = 0; k < Game.SIZE; k++)
				{
					((Rectangle)(ref val))._002Ector(SPACE + j * num2 + SPACE, SPACE + k * num2 + SPACE, num2 - 2 * SPACE, num2 - 2 * SPACE);
					if (game.getField(j, k) == 1)
					{
						g.DrawImage(spr_cross, val);
					}
					if (game.getField(j, k) == 2)
					{
						g.DrawImage(spr_nol, val);
					}
				}
			}
		}

		public bool getClickIJ(int x, int y, int w, int h, ref int celli, ref int cellj)
		{
			int num = Math.Min(w, h);
			int num2 = (num - 2 * SPACE) / Game.SIZE;
			Rectangle val = default(Rectangle);
			for (int i = 0; i < Game.SIZE; i++)
			{
				for (int j = 0; j < Game.SIZE; j++)
				{
					((Rectangle)(ref val))._002Ector(SPACE + i * num2 + SPACE, SPACE + j * num2 + SPACE, num2 - 2 * SPACE, num2 - 2 * SPACE);
					if (((Rectangle)(ref val)).get_X() < x && ((Rectangle)(ref val)).get_Y() < y && x < ((Rectangle)(ref val)).get_X() + ((Rectangle)(ref val)).get_Width() && y < ((Rectangle)(ref val)).get_Y() + ((Rectangle)(ref val)).get_Height())
					{
						celli = i;
						cellj = j;
						return true;
					}
				}
			}
			return false;
		}
	}
}
