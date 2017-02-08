import java.awt.Polygon;
import java.awt.Rectangle;
import java.awt.geom.Area;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

public class PolygonMatcherTrainer {
	private Map<String, List<Building>> buildingsPerImage;
	private final int numThreads = Math.max(1, Runtime.getRuntime().availableProcessors() * 95 / 100);
	private final int maxTrees = 60;
	private List<float[]> polyMatchFeatures = new ArrayList<float[]>();
	private List<Float> polyMatchValues = new ArrayList<Float>();
	private RandomForestPredictor buildingPredictor, borderPredictor, distPredictor;

	public static void main(String[] args) {
		File trainingCsv = new File("../competition1/spacenet_TrainData/vectordata/summarydata/AOI_1_RIO_polygons_solution_3band.csv");
		File folder3band = new File("../competition1/spacenet_TrainData/3band");
		File folder8band = new File("../competition1/spacenet_TrainData/8band");
		File rfBuilding = new File("out/rfBuilding.dat");
		File rfBorder = new File("out/rfBorder.dat");
		File rfDist = new File("out/rfDist.dat");
		File rfPolyMatch = new File("out/rfPolyMatch.dat");
		rfPolyMatch.getParentFile().mkdirs();
		new PolygonMatcherTrainer().train(trainingCsv, folder3band, folder8band, rfBuilding, rfBorder, rfDist, rfPolyMatch);
	}

	public void train(File trainingCsv, File folder3band, File folder8band, File rfBuilding, File rfBorder, File rfDist, File rfPolyMatch) {
		buildingsPerImage = Util.readBuildingsCsv(trainingCsv);
		Util.splitImages(buildingsPerImage, new int[] {60,30,10}, 1);
		loadPredictors(rfBuilding, rfBorder, rfDist);
		processImages(folder3band, folder8band);
		buildRandomForest(rfPolyMatch);
	}

	private void buildRandomForest(File rfPolyMatch) {
		try {
			System.err.println("Building Random Forest");
			long t = System.currentTimeMillis();
			float[] values = new float[polyMatchValues.size()];
			for (int i = 0; i < values.length; i++) {
				values[i] = polyMatchValues.get(i).floatValue();
			}
			polyMatchValues.clear();
			float[][] features = new float[polyMatchFeatures.get(0).length][values.length];
			for (int i = 0; i < values.length; i++) {
				float[] a = polyMatchFeatures.get(i);
				for (int j = 0; j < a.length; j++) {
					features[j][i] = a[j];
				}
			}
			polyMatchFeatures.clear();
			RandomForestBuilder.train(features, values, maxTrees, rfPolyMatch, numThreads);
			System.err.println("\t RF Poly Match: " + rfPolyMatch.length() + " bytes");
			System.err.println("\t  Elapsed Time: " + (System.currentTimeMillis() - t) + " ms");
			System.err.println();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void processImages(final File folder3band, final File folder8band) {
		try {
			System.err.println("Processing Images");
			long t = System.currentTimeMillis();
			final List<String> images = new ArrayList<String>(buildingsPerImage.keySet());
			Thread[] threads = new Thread[numThreads];
			for (int i = 0; i < numThreads; i++) {
				final int start = i;
				threads[i] = new Thread() {
					public void run() {
						for (int j = start; j < images.size(); j += numThreads) {
							//if (j!=23) continue;
							String image = images.get(j);
							File imageFile3band = new File(folder3band, "3band_" + image + ".tif");
							File imageFile8band = new File(folder8band, "8band_" + image + ".tif");
							List<Building> buildings = buildingsPerImage.get(image);
							processImage(imageFile3band, imageFile8band, buildings);
							if (start == 0) System.err.println("\t" + (j + 1) + "/" + images.size());
						}
					}
				};
				threads[i].start();
				threads[i].setPriority(Thread.MIN_PRIORITY);
			}
			for (int i = 0; i < numThreads; i++) {
				threads[i].join();
			}
			System.err.println("\tPoly Match Samples: " + polyMatchValues.size());
			System.err.println("\t      Elapsed Time: " + (System.currentTimeMillis() - t) + " ms");
			System.err.println();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void processImage(File imageFile3band, File imageFile8band, List<Building> groundTruthBuildings) {
		try {
			BufferedImage img3band = ImageIO.read(imageFile3band);
			BufferedImage img8band = ImageIO.read(imageFile8band);
			MultiChannelImage mci = new MultiChannelImage(img3band, img8band);
			double[][][] vals = Util.evalImage(mci, buildingPredictor, borderPredictor, distPredictor);
			double[][] buildingValues = vals[0];
			double[][] borderValues = vals[1];
			double[][] distValues = vals[2];

			List<Polygon> candidates = new ArrayList<Polygon>();
			for (int borderShift : PolygonFeatureExtractor.borderShifts) {
				for (double borderWeight : PolygonFeatureExtractor.borderWeights) {
					for (double cut : PolygonFeatureExtractor.buildingsCuts) {
						candidates.addAll(Util.findBuildings(buildingValues, borderValues, distValues, cut, borderWeight, PolygonFeatureExtractor.buildingsPolyBorder+borderShift, PolygonFeatureExtractor.buildingsPolyBorder2+borderShift));
					}
				}
			}
			for (Polygon polygon : candidates) {
				float value = (float) iou(polygon, groundTruthBuildings);
				float[] features = PolygonFeatureExtractor.getFeatures(mci, polygon, buildingValues, borderValues, distValues);
				synchronized (polyMatchValues) {
					polyMatchValues.add(value);
					polyMatchFeatures.add(features);
				}
			}
			/*
			int w = mci.width;
			int h = mci.height;
			BufferedImage img2 = new BufferedImage(w * 2, h, BufferedImage.TYPE_INT_BGR);
			Graphics2D g = img2.createGraphics();
			for (int y = 0; y < h; y++) {
			    for (int x = 0; x < w; x++) {
			        //int v = (int) (255 * buildingValue[y][x]) + 256 * 256 * (int) (255 * borderValue[y][x]);
					int v = Color.HSBtoRGB((float)(distValue[y][x]+10)/(float)(30+1) + 0.5f, 1, 1);
			        img2.setRGB(x, y, v);
			        img2.setRGB(x + w, y, img3band.getRGB(x, y));
			    }
			}
			g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
			g.setStroke(new BasicStroke(0.5f));
			//for (Building building : groundTruthBuildings) {
			    //Polygon poly = building.in.get(0);
			  //  g.setColor(Color.green);
			    //g.draw(poly);
			    //g.setColor(Color.yellow);
			    //g.draw(building.getRectPolygon());
			//}
			
			for (int i = 0; i < candidates.size(); i++) {
			    g.setColor(Color.green);
			    //g.draw(candidates.get(i).poly);
			    g.setColor(Color.yellow);
			    //g.draw(inner(candidates.get(i).poly));
			    //System.err.println(i + "\t" + eval(candidates.get(i).poly, w, h, buildingValue, borderValue, true));
			}
			
			//Polygon poly = new Polygon(new int[] {281,356,380,305}, new int[] {166,332,323,158}, 4);
			//g.setColor(Color.green);
			//g.draw(poly);
			//g.setColor(Color.yellow);
			//g.draw(inner(poly));
			//System.err.println(">>\t" + eval(poly, w, h, buildingValue, borderValue, true));
			
			g.dispose();
			new ImgViewer(img2);
			*/
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private double iou(Polygon poly, List<Building> buildings) {
		double best = 0;
		Area polyArea = new Area(poly);
		double polyAreaVal = Util.areaVal(polyArea);
		Rectangle polyRect = polyArea.getBounds();
		for (Building building : buildings) {
			Area buildingArea = building.getArea();
			Rectangle buildingRect = buildingArea.getBounds();
			if (!buildingRect.intersects(polyRect)) continue;
			Area interArea = new Area(polyArea);
			interArea.intersect(buildingArea);
			double a = Util.areaVal(interArea);
			double curr = a / (polyAreaVal + building.getAreaVal() - a);
			if (curr > best) best = curr;
		}
		return best;
	}

	private void loadPredictors(File rfBuilding, File rfBorder, File rfDist) {
		try {
			System.err.println("Loading Predictors");
			long t = System.currentTimeMillis();
			buildingPredictor = RandomForestPredictor.load(rfBuilding, -1);
			borderPredictor = RandomForestPredictor.load(rfBorder, -1);
			distPredictor = RandomForestPredictor.load(rfDist, -1);
			System.err.println("\t  Elapsed Time: " + (System.currentTimeMillis() - t) + " ms");
			System.err.println();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}